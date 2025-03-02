# Author: Xihan Li
# xihan.li@cs.ucl.ac.uk
# https://snowkylin.github.io
#
# Implementation of "Circuit Transformer: A Transformer That Preserves Logical Equivalence"
# https://openreview.net/forum?id=kpnW12Lm9p

from __future__ import annotations

import copy
import os
import pickle
import types
from collections import deque, Counter

import bitarray
import bitarray.util
import numpy as np
import scipy.special as special
import npn
import time
import json
import sys
import tracemalloc
import tensorflow as tf
import tf_keras as keras
from tensorflow_models import nlp
from circuit_transformer.tensorflow_transformer import Seq2SeqTransformer, CustomSchedule, masked_loss, masked_accuracy
from circuit_transformer.utils import *


def node_to_int(root: NodeWithInv, num_inputs: int):
    # zero for [PAD] that will be masked
    if root.is_leaf():
        return 2 + root.var * 2 + root.inverted  # [x_i] or [x_i_NOT]
    else:
        return 2 + num_inputs * 2 + root.inverted  # [AND] or [AND_NOT]


def int_to_node(token: int, num_inputs: int):
    if token == 0 or token == 1:  # [PAD] or [EOS]
        return False
    elif token < 2 + num_inputs * 2:  # [x_i] or [x_i_NOT]
        return NodeWithInv(Node((token - 2) // 2, None, None), token % 2)
    elif token < 2 + num_inputs * 2 + 2:  # [AND] or [AND_NOT]
        return NodeWithInv(Node(None, None, None), token % 2)
    else:
        raise ValueError(token)


def encode_aig(roots: list[NodeWithInv], num_inputs: int) -> (list[int], list[int]):
    def encode_aig_rec(root: NodeWithInv, seq_enc: list[int], cur_pos_enc: int, pos_enc: list[int]):
        seq_enc.append(node_to_int(root, num_inputs))
        pos_enc.append(cur_pos_enc)
        if not root.is_leaf():
            encode_aig_rec(root.left, seq_enc, (cur_pos_enc << 2) + 1, pos_enc)
            encode_aig_rec(root.right, seq_enc, (cur_pos_enc << 2) + 2, pos_enc)

    seq_enc, pos_enc = [], []
    assert len(roots) <= 2
    encode_aig_rec(roots[0], seq_enc, 1, pos_enc)
    if len(roots) == 2:
        encode_aig_rec(roots[1], seq_enc, 2, pos_enc)
    return seq_enc, pos_enc


def stack_to_encoding(tree_stack: list, root_id: int, max_tree_depth: int):
    assert len(tree_stack) <= max_tree_depth
    assert root_id >= 0
    encoding = np.zeros(max_tree_depth * 2, np.float32)
    for i, node in enumerate(reversed(tree_stack)):
        if i == 0:
            if node.left is None:  # the current node should be inserted on the left side
                encoding[i * 2] = 1.
            else:
                encoding[i * 2 + 1] = 1.
        else:
            if node.right is None:  # the current node is inserted on the left side
                encoding[i * 2] = 1.
            else:
                encoding[i * 2 + 1] = 1.
    encoding[len(tree_stack) * 2 + root_id] = 1.
    return encoding


def deref_node(root: Node, ref_dict: dict, context_nodes=None, verbose=0):
    if context_nodes is not None and root in context_nodes:
        return 0
    if root.is_leaf():
        return 0
    value = 1
    for child in [root.left.parent, root.right.parent]:
        if verbose > 1:
            print("ref %s (parent %s, %s) from %d to %d" %
                  (child, root, "left" if child is root.left.parent else "right", ref_dict[child], ref_dict[child] - 1))
        ref_dict[child] -= 1
        if ref_dict[child] == 0:
            value += deref_node(child, ref_dict, context_nodes, verbose)
    return value


class LogicNetworkEnv:
    def __init__(self,
                 tts,
                 num_inputs,
                 context_num_inputs=None,
                 input_tt=None,
                 init_care_set_tt=None,     # for the first output (which can be computed in advance) or for all the outputs (list[num_outputs])
                 max_tree_depth=32,
                 max_inference_tree_depth=16,
                 max_inference_reward=None,
                 max_length=None,
                 eos_id=1,
                 pad_id=0,
                 context_hash: set = None,
                 ffw = None,
                 and_always_available=False,    # for training
                 use_controllability_dont_cares=True,   # Patterns that cannot happen at inputs to a network node.
                 tts_compressed=None,                   # must specify when `use_controllability_dont_cares` = False, the truth table of 2^num_inputs corresponding to the "local" aig
                 verbose=0,
                 ):
        # assert len(tts) == 2
        self.num_outputs = len(tts)
        self.num_inputs = num_inputs
        self.context_num_inputs = context_num_inputs if context_num_inputs is not None else num_inputs
        self.tts_bitarray = tts
        self.init_care_set_tt = init_care_set_tt if init_care_set_tt is not None else bitarray.util.ones(2 ** self.context_num_inputs)
        self.ffw = ffw
        self.roots = []
        self.tokens = []
        self.positional_encodings = []
        self.action_masks = []
        self.is_finished = False
        self.gen_eos = False
        self.tree_stack = []
        # self.tt_hash = {}
        # self.tt_cache = {}
        self.context_hash = context_hash
        self.t = 0
        self.max_length = max_length
        self.rewards = []
        self.EOS = eos_id
        self.PAD = pad_id
        self.max_tree_depth = max_tree_depth        # for positional encoding
        self.max_inference_tree_depth = max_inference_tree_depth    # for pruning failed circuits
        self.max_inference_reward = max_inference_reward
        self.and_always_available = and_always_available
        self.use_controllability_dont_cares = use_controllability_dont_cares
        self.unfinished_penalty = -10
        self.verbose = verbose
        if input_tt is None:
            self.input_tt_bitarray = compute_input_tt(self.context_num_inputs)
        else:
            self.input_tt_bitarray = input_tt
        self.tt_cache_bitarray = {Node(i // 2, None, None): v
                                  for i, v in enumerate(self.input_tt_bitarray) if i % 2 == 0}
        self.tt_hash_bitarray = {v.tobytes(): node for node, v in self.tt_cache_bitarray.items()}
        self.vocab_size = 2 + num_inputs * 2 + 2
        self.ref_dict = {k: 1 for k in self.tt_cache_bitarray.keys()}
        self.context_nodes = set()
        self.context_records = dict()

        if self.use_controllability_dont_cares:
            self.initialize_care_set_tt()
        else:
            assert tts_compressed is not None
            assert init_care_set_tt is None
            self.compress_indices = None
            self.input_tt_bitarray_compressed = compute_input_tt(len(self.input_tt_bitarray) // 2)
            self.tts_bitarray_compressed = tts_compressed
            self.tt_cache_bitarray_compressed = {Node(i // 2, None, None): v
                                                 for i, v in enumerate(self.input_tt_bitarray_compressed) if i % 2 == 0}

        self.action_masks.append(self.gen_action_mask())

    @property
    def cur_root_id(self):
        return len(self.roots) - (1 if len(self.tree_stack) > 0 else 0)

    @property
    def cumulative_reward(self):
        return sum(self.rewards)

    @property
    def min_cumulative_reward(self):
        res = np.iinfo(int).max
        cumulative_reward = 0
        for r in self.rewards:
            cumulative_reward += r
            res = min(res, cumulative_reward)
        return res

    @property
    def success(self):
        return self.gen_eos

    def initialize_care_set_tt(self):       # both controllability and observability don't cares
        if self.cur_root_id == 0:
            self.care_set_tt = self.init_care_set_tt[self.cur_root_id] if isinstance(self.init_care_set_tt, list) else self.init_care_set_tt
        else:
            if self.ffw is not None:
                new_inputs = get_inputs_rec(self.roots)
                modified_list = []
                for extracted_input, orig_node in self.ffw.input_mapping.items():
                    if extracted_input.var in new_inputs:
                        for new_input_with_inv in new_inputs[extracted_input.var]:
                            modified_list.append((new_input_with_inv, new_input_with_inv.parent))
                            new_input_with_inv.parent = orig_node
                for new_output, output in zip(self.roots, self.ffw.outputs):
                    for node_with_inv in self.ffw.parent.fanout_dict[output].keys():
                        node_with_inv.parent = new_output.parent
                        if new_output.inverted:
                            node_with_inv.inverted = not node_with_inv.inverted
                if not detect_circle(self.ffw.parent.outputs):
                    self.care_set_tt = self.ffw.parent.compute_care_set(self.ffw.outputs[self.cur_root_id])
                else:
                    self.care_set_tt = bitarray.util.ones(2 ** self.context_num_inputs)
                for new_output, output in zip(self.roots, self.ffw.outputs):
                    for node_with_inv in self.ffw.parent.fanout_dict[output].keys():
                        node_with_inv.parent = output
                        if new_output.inverted:
                            node_with_inv.inverted = not node_with_inv.inverted
                for new_input_with_inv, parent in modified_list:
                    new_input_with_inv.parent = parent
            elif isinstance(self.init_care_set_tt, list):
                self.care_set_tt = self.init_care_set_tt[self.cur_root_id]

        a = bytearray()
        len_care_set = self.care_set_tt.count()
        assert len(self.input_tt_bitarray) // 2 <= 8  # one byte
        for i, tt in enumerate(self.input_tt_bitarray):
            if i % 2 == 0:
                a.extend(tt[self.care_set_tt].unpack(one=(1 << (i // 2)).to_bytes(1, 'big')))
        a_np = np.frombuffer(a, dtype=np.uint8).reshape(len(self.input_tt_bitarray) // 2, len_care_set)
        a_np = np.sum(a_np, axis=0, dtype=np.uint8)
        a_np_unique, self.compress_indices = np.unique(a_np, return_index=True)

        if self.verbose > 1:
            a = bytearray()
            for i, tt in enumerate(self.input_tt_bitarray):
                if i % 2 == 0:
                    a.extend(tt.unpack(one=(1 << (i // 2)).to_bytes(1, 'big')))
            a_np_ = np.frombuffer(a, dtype=np.uint8).reshape(len(self.input_tt_bitarray) // 2, len(self.care_set_tt))
            a_np_ = np.sum(a_np_, axis=0, dtype=np.uint8)
            a_np_unique_, self.compress_indices_no_care_set = np.unique(a_np_, return_index=True)
            if len(self.compress_indices_no_care_set) > len(self.compress_indices):
                print("care set size: %d, without care set: %d, with care set: %d" %
                      (self.care_set_tt.count(), len(self.compress_indices_no_care_set), len(self.compress_indices)))

        self.compress_indices = list(self.compress_indices)
        a_bitarray_unique = [bitarray.bitarray() for _ in a_np_unique]
        for a_bitarray_i, a_np_i in zip(a_bitarray_unique, a_np_unique):
            a_bitarray_i.frombytes(a_np_i.tobytes())

        if len(a_bitarray_unique) == 0:
            self.input_tt_bitarray_compressed = [bitarray.bitarray() for _ in self.input_tt_bitarray]
        else:
            self.input_tt_bitarray_compressed = []
            for i, a_tuple in enumerate(zip(*a_bitarray_unique)):
                if i < 8 - len(self.input_tt_bitarray) // 2:
                    continue
                a_bitarray = bitarray.bitarray(a_tuple)
                self.input_tt_bitarray_compressed.extend([~a_bitarray, a_bitarray])
            self.input_tt_bitarray_compressed.reverse()
        # self.input_tt_bitarray_compressed_ = [bitarray.bitarray(_) for _ in zip(*a_bitarray_unique)]
        tts_care_set = [tt[self.care_set_tt] for tt in self.tts_bitarray]
        self.tts_bitarray_compressed = [bitarray.bitarray([tt[i] for i in self.compress_indices]) for tt in tts_care_set]
        self.tt_cache_bitarray_compressed = {Node(i // 2, None, None): v
                                             for i, v in enumerate(self.input_tt_bitarray_compressed) if i % 2 == 0}

    def compress(self, tt):
        return (tt[self.care_set_tt])[self.compress_indices]

    def step(self, token):
        token = int(token)
        self.positional_encodings.append(stack_to_encoding(self.tree_stack, self.cur_root_id, self.max_tree_depth))
        if token == self.EOS:
            self.is_finished = True
            if self.gen_eos:
                reward, done = 0, True
            else:
                reward, done = self.unfinished_penalty, True
        elif self.is_finished:
            assert token == self.PAD
            reward, done = 0, True
        elif not self.is_finished and self.t >= self.max_length - 1:  # reached the last step but still not finished
            reward, done = self.unfinished_penalty, True
        else:
            node = int_to_node(token, self.num_inputs)
            if len(self.tree_stack) == 0:
                self.roots.append(node)
            else:
                # insert node into the tree
                if self.tree_stack[-1].left is None:
                    self.tree_stack[-1].left = node
                else:
                    self.tree_stack[-1].right = node
            self.ref_dict[node.parent] = 1
            # calculate reward
            reward = 0 if node.is_leaf() else -1
            done = False
            # update stack
            if node.is_leaf():
                self.tree_stack.append(node)
                while len(self.tree_stack) > 0 and (self.tree_stack[-1].is_leaf() or (
                        self.tree_stack[-1].left is not None and self.tree_stack[-1].right is not None)):
                    old_node = copy.copy(self.tree_stack[-1])
                    old_node.inverted = False
                    tt_bitarray = compute_tt(old_node, input_tt=self.input_tt_bitarray, cache=self.tt_cache_bitarray)
                    tt_not_bitarray = ~tt_bitarray
                    tt = tt_bitarray.tobytes()
                    tt_not = tt_not_bitarray.tobytes()
                    create_new_hash = True
                    if tt in self.tt_hash_bitarray or tt_not in self.tt_hash_bitarray:
                        inverted = tt_not in self.tt_hash_bitarray
                        new_node = self.tt_hash_bitarray[tt_not if inverted else tt]
                        if self.ref_dict[new_node] > 0:     # use existing node to replace self.tree_stack[-1]
                            create_new_hash = False
                            new_node_with_inv = NodeWithInv(new_node, (not inverted) if self.tree_stack[-1].inverted else inverted)
                            self.tt_cache_bitarray[new_node_with_inv] = tt_not_bitarray if self.tree_stack[-1].inverted else tt_bitarray
                            if self.use_controllability_dont_cares:
                                self.tt_cache_bitarray_compressed[new_node_with_inv] = self.compress(self.tt_cache_bitarray[new_node_with_inv])
                            else:
                                tt_bitarray_compressed = compute_tt(old_node, input_tt=self.input_tt_bitarray_compressed, cache=self.tt_cache_bitarray_compressed)
                                self.tt_cache_bitarray_compressed[new_node_with_inv] = (~tt_bitarray_compressed) if self.tree_stack[-1].inverted else tt_bitarray_compressed
                            if len(self.tree_stack) > 1:
                                if self.tree_stack[-2].left is self.tree_stack[-1]:
                                    self.tree_stack[-2].left = new_node_with_inv
                                else:
                                    self.tree_stack[-2].right = new_node_with_inv
                            else:
                                self.roots[self.cur_root_id] = new_node_with_inv
                            self.ref_dict[new_node] += 1
                            self.ref_dict[self.tree_stack[-1].parent] -= 1
                            v1 = deref_node(self.tree_stack[-1].parent, self.ref_dict, self.context_nodes)
                            reward += v1
                    if create_new_hash:
                        self.tt_hash_bitarray[tt_bitarray.tobytes()] = self.tree_stack[-1].parent
                        self.tt_cache_bitarray[self.tree_stack[-1]] = tt_not_bitarray if self.tree_stack[-1].inverted else tt_bitarray
                        if self.use_controllability_dont_cares:
                            self.tt_cache_bitarray_compressed[self.tree_stack[-1]] = self.compress(self.tt_cache_bitarray[self.tree_stack[-1]])
                        else:
                            tt_bitarray_compressed = compute_tt(self.tree_stack[-1],
                                                                               input_tt=self.input_tt_bitarray_compressed,
                                                                               cache=self.tt_cache_bitarray_compressed)
                            self.tt_cache_bitarray_compressed[self.tree_stack[-1]] = tt_bitarray_compressed
                        if self.context_hash is not None and (tt in self.context_hash or tt_not in self.context_hash):
                            v1 = deref_node(self.tree_stack[-1].parent, self.ref_dict, self.context_nodes)
                            self.context_nodes.add(self.tree_stack[-1].parent)
                            self.context_records[self.tree_stack[-1].parent] = tt
                            reward += v1
                    self.tree_stack.pop()
                if len(self.tree_stack) == 0 and len(self.roots) == self.num_outputs:
                    self.gen_eos = True  # next token should be EOS
                    done = True
            else:
                self.tree_stack.append(node)
        self.tokens.append(token)
        self.t += 1
        self.rewards.append(reward)
        if len(self.tree_stack) == 0 and self.cur_root_id < self.num_outputs and self.use_controllability_dont_cares:
            self.initialize_care_set_tt()

        self.action_masks.append(self.gen_action_mask())
        return reward, done

    def gen_action_mask(self):
        action_mask_ba = bitarray.util.zeros(self.vocab_size)
        cur_node = None if len(self.tree_stack) == 0 else self.tree_stack[-1]
        action_mask_ba[self.EOS] = cur_node is None and not self.is_finished and len(self.roots) == self.num_outputs
        action_mask_ba[self.PAD] = self.is_finished
        if not self.is_finished and not self.gen_eos and \
                (self.max_inference_reward is None or self.cumulative_reward >= self.max_inference_reward):
            # insert the node into the tree
            # var = -2 means not determined (check all possibilities)
            node = NodeWithInv(parent=Node(var=-2, left=None, right=None), inverted=False)
            if cur_node is None:
                is_root = True
            else:
                is_root = False
                if cur_node.left is None:
                    cur_node.left = node
                else:
                    cur_node.right = node
            has_conflict_ba, completeness_ba = check_conflict(self.tree_stack, self.tts_bitarray_compressed[self.cur_root_id],
                                                              self.input_tt_bitarray_compressed, self.tt_cache_bitarray_compressed)
            value_action_mask_ba = ~has_conflict_ba
            action_mask_ba[2: 2 + len(value_action_mask_ba)] = value_action_mask_ba
            if self.and_always_available:
                action_mask_ba[2 + self.num_inputs * 2: 4 + self.num_inputs * 2] = bitarray.bitarray('11')
            else:
                action_mask_ba[2 + self.num_inputs * 2: 4 + self.num_inputs * 2] = bitarray.bitarray('00') \
                    if (value_action_mask_ba & completeness_ba).any() or len(self.tree_stack) >= self.max_inference_tree_depth - 2 \
                    else bitarray.bitarray('11')

            # remove the node from the tree
            if not is_root:
                if cur_node.right is None:
                    cur_node.left = None
                else:
                    cur_node.right = None
        if not action_mask_ba.any():
            action_mask_ba[self.EOS] = True
        return np.array(action_mask_ba.tolist(), dtype=bool)


class MCTSNode:
    INIT_MAX_VALUE = -1000

    def __init__(self, parent, t, action, prob=None, info=None, puct_explore_ratio=1.):
        self.t = t
        self.parent = parent
        self.action = action
        self.explored = False
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.total_value = 0
        self.info = info
        self.v = None
        self.prob = prob
        self.max_value = self.INIT_MAX_VALUE
        self.puct_explore_ratio = puct_explore_ratio

    @property
    def value(self):  # Q
        return self.total_value / self.visits if self.visits != 0 else 100

    @property
    def puct(self):
        return self.value + self.puct_explore_ratio * self.prob * np.sqrt(self.parent.visits) / (1 + self.visits)

    def __repr__(self):     # sum reward: from the root to the end, value: future reward from (excluding) the current node to the end
        repr = "(%s%s, visits: %d, avg sum reward: %.2f, max sum reward: %d, value: %s, seq: %s)" % \
               (self.action, " (Done)" if self.info['done'] else "", self.visits, self.value, self.max_value, self.v, self.info['env'].tokens)
        if self.prob is not None:
            repr = repr[:-1] + ", prob: %.2f, puct: %.2f)" % (self.prob, self.puct)
        return repr


def ucb(node: MCTSNode):
    return node.value + np.sqrt(np.log(node.parent.visits) / node.visits)


class CircuitTransformer:
    def __init__(self,
                 num_inputs=8,
                 embedding_width=512,
                 num_layers=12,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 max_tree_depth=32,
                 max_seq_length=200,
                 inference_batch_size=512,
                 eos_id=1,
                 pad_id=0,
                 verbose=0,
                 mixed_precision=True,
                 ckpt_path=None,
                 batch_size=None,
                 add_action_mask_to_input=False,
                 policy_temperature_in_mcts=1.
                 ):
        self.num_inputs = num_inputs
        self.vocab_size = 2 + 2 * self.num_inputs + 2
        self.embedding_width = embedding_width
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_tree_depth = max_tree_depth
        self.max_seq_length = max_seq_length
        self.inference_batch_size = inference_batch_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.verbose = verbose
        self.ckpt_path = ckpt_path
        self.add_action_mask_to_input = add_action_mask_to_input
        self.policy_temperature_in_mcts = policy_temperature_in_mcts

        # https://www.tensorflow.org/guide/mixed_precision
        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        self._transformer = self._get_tf_transformer()

        if self.ckpt_path is not None:
            self.load(self.ckpt_path)

        @tf.function(reduce_retracing=True)
        def _transformer_inference_graph(self, inputs, return_kv_cache=False, return_last_token=False):
            policy, cache = self._transformer(inputs, return_kv_cache=return_kv_cache, return_last_token=return_last_token)
            return policy, cache

        def _transformer_inference(self, inputs, return_kv_cache=False, return_last_token=False):
            policy, cache = _transformer_inference_graph(self, inputs, return_kv_cache=return_kv_cache, return_last_token=return_last_token)
            return policy.numpy(), cache

        self._transformer_inference = types.MethodType(_transformer_inference, self)
        self._transformer.return_cache = True
        self.use_kv_cache = True
        self.input_tt = compute_input_tt(self.num_inputs)

    def _get_tf_transformer(self):
        return Seq2SeqTransformer(
            enc_vocab_size=self.vocab_size,
            dec_vocab_size=self.vocab_size,
            embedding_width=self.embedding_width,
            encoder_layer=nlp.models.TransformerEncoder(
                num_layers=self.num_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size
            ),
            decoder_layer=nlp.models.TransformerDecoder(
                num_layers=self.num_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size
            ),
            eos_id=self.eos_id,
            max_tree_depth=self.max_tree_depth,
            add_action_mask_to_inputs=self.add_action_mask_to_input
        )

    def _copy_env(self, env: LogicNetworkEnv | list[LogicNetworkEnv]):
        if isinstance(env, list):
            return [self._copy_env(e) for e in env]
        else:
            context_hash, tts_bitarray, input_tt_bitarray, input_tt_bitarray_compressed, ffw = \
                env.context_hash, env.tts_bitarray, env.input_tt_bitarray, env.input_tt_bitarray_compressed, env.ffw
            env.context_hash, env.tts, env.input_tt, env.input_tt_bitarray_compressed, env.ffw = None, None, None, None, None
            res = copy.deepcopy(env)
            env.context_hash, env.tts_bitarray, env.input_tt_bitarray, env.input_tt_bitarray_compressed, env.ffw = \
                context_hash, tts_bitarray, input_tt_bitarray, input_tt_bitarray_compressed, ffw
            res.context_hash, res.tts_bitarray, res.input_tt_bitarray, res.input_tt_bitarray_compressed, res.ffw = \
                context_hash, tts_bitarray, input_tt_bitarray, input_tt_bitarray_compressed, ffw
            return res

    def _batch_estimate_policy(self, envs: list[LogicNetworkEnv], src_tokens, src_pos_enc, src_action_mask, action_masks, cache):
        start_time = time.time()
        indices = [len(env.tokens) for env in envs]
        max_token_length = np.max(indices) + 1
        tgt_tokens = np.stack(
            [np.array(env.tokens + [0] * (max_token_length - len(env.tokens)), dtype=np.int32) for env in envs],
            axis=0)
        tgt_pos_enc = np.stack(
            [np.stack(
                env.positional_encodings + [np.zeros(self.max_tree_depth * 2, dtype=np.float32)] * (
                        max_token_length - len(env.tokens)),
                axis=0) for env in envs]
            , axis=0)
        tgt_action_mask = np.stack(
            [np.concatenate([np.stack(env.action_masks),
                            np.ones((max_token_length - len(env.action_masks), self.vocab_size), dtype=bool)], axis=0)
            for env in envs]
        )
        inputs = {'inputs': src_tokens, 'enc_pos_encoding': src_pos_enc,
                  'targets': tgt_tokens, 'dec_pos_encoding': tgt_pos_enc,
                  'enc_action_mask': src_action_mask, 'dec_action_mask': tgt_action_mask}
        if cache is not None:
            inputs['cache'] = cache
        policy, cache = self._transformer_inference(inputs)
        policy = np.stack([policy_i[j] for policy_i, j in zip(policy, indices)], axis=0)
        if self.verbose > 0:
            print("policy estimation time: %.2f" % (time.time() - start_time))
        return special.softmax(np.where(action_masks, policy / self.policy_temperature_in_mcts, np.finfo(np.float32).min), axis=1), cache

    def _batch_estimate_v_value_via_simulation_kvcache(self, envs: list[LogicNetworkEnv], src_tokens, src_pos_enc, src_action_mask,
                                                       max_inference_seq_length, cache=None, num_leaf_parallelization=1):
        total_time = time.time()
        envs = self._copy_env(envs)
        copy_time = time.time() - total_time
        batch_size = len(envs)
        v = np.zeros(len(envs), dtype=int)
        inputs = {'inputs': src_tokens, 'enc_pos_encoding': src_pos_enc, 'enc_action_mask': src_action_mask, 'cache': cache}
        targets = np.zeros((batch_size, 1), dtype=np.int32)
        dec_pos_encoding = np.zeros((batch_size, 1, self.max_tree_depth * 2), dtype=np.float32)
        if cache is not None and num_leaf_parallelization > 1:
            cache['encoder_outputs'] = np.concatenate([cache['encoder_outputs']] * num_leaf_parallelization, axis=0)
        transformer_time = 0.
        action_mask_time = 0.
        step_time = 0.
        for i in range(max_inference_seq_length):
            # print(i)
            inputs['targets'], inputs['dec_pos_encoding'] = targets, dec_pos_encoding

            # generate action mask
            # action_masks = np.stack([e.gen_action_mask() for e in envs], axis=0)
            start_time = time.time()
            action_masks = [e.action_masks[i] for e in envs]# [e.gen_action_mask() if i == e.t else np.ones(self.vocab_size, dtype=bool) for e in envs]
            action_masks = np.stack(action_masks, axis=0)
            inputs['dec_action_mask'] = np.expand_dims(action_masks, axis=1)
            action_mask_time += time.time() - start_time

            start_time = time.time()
            policy, cache = self._transformer_inference(inputs, return_kv_cache=True, return_last_token=True)
            inputs['cache'] = cache
            transformer_time += time.time() - start_time

            start_time = time.time()
            tokens = np.argmax(policy, axis=1)
            tokens = [token if i == e.t else e.tokens[i] for token, e in zip(tokens, envs)]

            rewards, dones = zip(*[e.step(token) if i == e.t else (0, False) for token, e in zip(tokens, envs)])
            step_time += time.time() - start_time

            dec_pos_encoding = [e.positional_encodings[i] for e in envs]

            v += np.array(rewards)
            if all([e.is_finished for e in envs]):
                break

            pos_encodings = np.expand_dims(np.stack(dec_pos_encoding, axis=0),
                                           axis=1)  # [batch_size, 1, max_tree_depth * 2]
            targets_new = np.expand_dims(tokens, axis=1)
            if self.use_kv_cache:
                targets = targets_new
                dec_pos_encoding = pos_encodings
            else:
                targets = np.concatenate([targets, targets_new], axis=1)
                dec_pos_encoding = np.concatenate([dec_pos_encoding, pos_encodings], axis=1)
        if cache is not None:
            cache['kv_cache'] = None
            if num_leaf_parallelization > 1:
                cache['encoder_outputs'] = cache['encoder_outputs'][:(batch_size // num_leaf_parallelization)]
        if self.verbose > 0:
            print("simulation time: total %f, copy %.2f, step %.2f, transformer %.2f, action mask %.2f; #(steps) = %d" %
                  (time.time() - total_time, copy_time, step_time, transformer_time, action_mask_time, i))
        return v.tolist(), [e.success for e in envs], cache

    def _batch_MCTS_policy_with_leaf_parallelization(self, envs: list[LogicNetworkEnv], num_leaf_parallelizations=8, num_playouts=100, max_inference_seq_length=None,
                           src_tokens=None, src_pos_enc=None, src_action_mask=None,
                           roots=None, orig_aigs_size=None, puct_explore_ratio=1.):
        def update_done_node(node: MCTSNode, root_id):
            if node is not roots[root_id]:  # update done info to avoid unnecessary search
                done_node = node.parent
                while np.all([c.info['done'] and c.explored for c in done_node.children]):
                    if self.verbose > 1:
                        print("node done:", done_node, done_node.children)
                    done_node.info['done'] = True
                    done_node = done_node.parent
                    if done_node is None:
                        break

        if max_inference_seq_length is None:
            max_inference_seq_length = self.max_seq_length

        envs = self._copy_env(envs)
        for env in envs:
            env.max_length = max_inference_seq_length

        src_tokens_parallel = np.concatenate([src_tokens] * num_leaf_parallelizations, axis=0)
        src_pos_enc_parallel = np.concatenate([src_pos_enc] * num_leaf_parallelizations, axis=0)
        src_action_mask_parallel = np.concatenate([src_action_mask] * num_leaf_parallelizations, axis=0)

        num_envs = len(envs)
        t = 0
        cache = None
        virtual_loss = -50
        if roots is None:
            roots = [MCTSNode(None, t, None, info={'env': env, 'reward': None, 'done': None, 'rollout_success': None},
                              puct_explore_ratio=puct_explore_ratio)for env in envs]  # info = (state, reward, done)

        for i in range(num_playouts):
            playout_time = time.time()
            nodes = [copy.copy(roots) for _ in range(num_leaf_parallelizations)]
            sum_rewards = [[root.info['env'].cumulative_reward for root in roots] for _ in range(num_leaf_parallelizations)]

            # selection
            expansion_time = time.time()
            largest_node_list = [[] for _ in range(num_leaf_parallelizations)]
            for l in range(num_leaf_parallelizations):
                largest_node_set = set(range(num_envs))
                while len(largest_node_set) > 0:
                    for j in largest_node_set:
                        node = roots[j]
                        sum_rewards[l][j] = roots[j].info['env'].cumulative_reward
                        while node.children:
                            not_done_children = [(c_id, c) for c_id, c in enumerate(node.children) if
                                                 not c.info['done'] or not c.explored]
                            if len(not_done_children) == 0:
                                # assert node is root
                                if self.verbose > 1:
                                    print(i, nodes[l][j], "all the children are done and explored!")
                                break
                            select_id = np.argmax([c.puct for c_id, c in not_done_children])
                            node.visits += 1
                            node.total_value += virtual_loss
                            node = node.children[not_done_children[select_id][0]]
                            sum_rewards[l][j] += node.info['reward']
                        nodes[l][j] = node

                    # expansion
                    action_masks = np.stack([node.info['env'].action_masks[-1] for node in nodes[l]], axis=0)
                    policies, cache = self._batch_estimate_policy([node.info['env'] for node in nodes[l]], src_tokens, src_pos_enc, src_action_mask,
                                                                  action_masks, cache)

                    new_largest_node_set = largest_node_set.copy()
                    for j, node in enumerate(nodes[l]):
                        if j in largest_node_set:
                            if node.explored:
                                new_action_list = np.where(action_masks[j])[0]
                                for token in new_action_list:
                                    e = self._copy_env(node.info['env'])
                                    reward, done = e.step(token)
                                    child = MCTSNode(node, node.t + 1, token, prob=policies[j][token],
                                                     info={'env': e, 'reward': reward, 'done': done, 'rollout_success': None},
                                                     puct_explore_ratio=puct_explore_ratio)
                                    node.children.append(child)
                                sorted_ids = np.argsort([c.puct for c in node.children])
                                node.visits += 1
                                node.total_value += virtual_loss
                                if len(sorted_ids) > 1:
                                    select_id, largest_id = sorted_ids[-2:]
                                    largest_node = node.children[largest_id]
                                    largest_node.explored = True
                                    largest_node.visits += 1
                                    largest_node.total_value += virtual_loss
                                    largest_node.v = virtual_loss
                                    largest_node.info['rollout_success'] = True
                                    largest_node.info['raw_value'] = virtual_loss
                                    largest_node_list[l].append((largest_node, node))
                                else:
                                    select_id = sorted_ids[-1]
                                    new_largest_node_set.remove(j)
                                    node = node.children[select_id]
                                    sum_rewards[l][j] += node.info['reward']
                            else:
                                node.explored = True
                                new_largest_node_set.remove(j)

                            if j not in new_largest_node_set:
                                node.visits += 1
                                node.total_value += virtual_loss
                                node.v = virtual_loss
                                node.info['rollout_success'] = True
                                node.info['raw_value'] = virtual_loss

                                nodes[l][j] = node
                    largest_node_set = new_largest_node_set

            expansion_time = time.time() - expansion_time

            # rollout (simulation)
            simulation_time = time.time()
            vs, rollout_success, cache = self._batch_estimate_v_value_via_simulation_kvcache(
                sum([[node.info['env'] for node in nodes_i] for nodes_i in nodes], start=[]),
                src_tokens_parallel, src_pos_enc_parallel, src_action_mask_parallel,
                max_inference_seq_length, cache, num_leaf_parallelizations)

            vs = np.array(vs).reshape((num_leaf_parallelizations, num_envs))
            rollout_success = np.array(rollout_success).reshape((num_leaf_parallelizations, num_envs))
            simulation_time = time.time() - simulation_time

            paths = [[[] for _ in range(num_envs)] for _ in range(num_leaf_parallelizations)]
            for l in range(num_leaf_parallelizations):
                for j, node in enumerate(nodes[l]):
                    if orig_aigs_size is not None and (not rollout_success[l][j] or vs[l][j] < -orig_aigs_size[j] - sum_rewards[l][j]):
                        node.v = -orig_aigs_size[j] - sum_rewards[l][j]
                    else:
                        node.v = vs[l][j]
                    sum_rewards[l][j] += node.v
                    node.explored = True
                    node.sum_reward = sum_rewards[l][j]
                    node.info['rollout_success'] = rollout_success[l][j]
                    node.info['raw_value'] = vs[l][j]
                    update_done_node(node, j)

                # backpropagate
                for j, node in enumerate(nodes[l]):
                    while node is not None:
                        if node.action is not None:
                            paths[l][j].append(node.action)
                        node.total_value += sum_rewards[l][j] - virtual_loss
                        if self.verbose > 1:
                            if sum_rewards[l][j] > node.max_value and node.max_value != MCTSNode.INIT_MAX_VALUE and node is roots[j]:
                                print("root %d node max value updated, from %d to %d!" % (j, node.max_value, sum_rewards[l][j]))
                        node.max_value = max(node.max_value, sum_rewards[l][j])
                        node = node.parent

                for largest_node, node in largest_node_list[l]:
                    largest_node.explored = True  # so #(explored nodes) >= num_rollouts
                    largest_node.v = node.v - largest_node.info['reward']
                    largest_node.max_value = node.max_value
                    largest_node.sum_reward = node.sum_reward
                    largest_node.info['rollout_success'] = node.info['rollout_success']
                    largest_node.info['raw_value'] = node.info['raw_value'] - largest_node.info['reward']
                    largest_node.info['derived_from_parent'] = True
                    while largest_node is not None:
                        largest_node.total_value += node.sum_reward - virtual_loss   # node.v
                        largest_node = largest_node.parent

            if self.verbose > 0:
                for l in range(num_leaf_parallelizations):
                    print([(p, s) for p, s in zip(paths[l], sum_rewards[l])])
                print([r.max_value for r in roots])
                print("playout %d - total time %.2f, expansion time %.2f, simulation time %.2f" %
                      (i, time.time() - playout_time, expansion_time, simulation_time))

        best_action_seqs = []
        best_child_seqs = []
        for j, root in enumerate(roots):
            best_action_seq, best_child_seq = [], []
            while root.children:
                action_list = [(c, c.max_value, c.puct) for c in root.children]
                action_list.sort(key=lambda x: (x[1], x[2]), reverse=True)
                root = action_list[0][0]
                if not root.explored:
                    break
                best_action_seq.append(root.action)
                best_child_seq.append(root)
            best_action_seqs.append(best_action_seq)
            best_child_seqs.append(best_child_seq)
        return best_action_seqs, [b[0] for b in best_child_seqs]

    def _encode_postprocess(self, seq_enc: list[int], pos_enc: list[int]):
        seq_enc.append(self.eos_id)
        if self.verbose > 0 and len(seq_enc) > self.max_seq_length:
            print("Warning: seq_enc length %d > max seq length (%d)" % (len(seq_enc), self.max_seq_length))
        seq_enc, pos_enc = seq_enc[:self.max_seq_length], pos_enc[:self.max_seq_length]
        pos_enc = np.stack(
            [list(reversed(npn.int_to_tt(pos_enc_i, base_2_log(self.max_tree_depth) + 1))) for pos_enc_i in pos_enc],
            axis=0)  # 2 ^ 6 = 64 == max_tree_depth * 2
        seq_enc = np.array(seq_enc + [0] * (self.max_seq_length - len(seq_enc)), dtype=np.int32)
        pos_enc = np.concatenate([pos_enc, np.zeros((self.max_seq_length - len(pos_enc), self.max_tree_depth * 2))],
                                 axis=0,
                                 dtype=np.float32)
        return seq_enc, pos_enc

    def load(self, ckpt_path):
        self._transformer.load_weights(ckpt_path)
        self.ckpt_path = ckpt_path

    def load_from_hf(self, hf_model_name="deepsyn_reinforced"):
        from huggingface_hub import hf_hub_download
        index_path = hf_hub_download(repo_id="snowkylin/circuit-transformer", filename="%s.index" % hf_model_name)
        data_path = hf_hub_download(repo_id="snowkylin/circuit-transformer", filename="%s.data-00000-of-00001" % hf_model_name)
        ckpt_path = index_path[:-6]
        print("ckeckpoint downloaded to %s" % ckpt_path)
        self._transformer.load_weights(ckpt_path)
        self.ckpt_path = index_path

    def generate_action_masks(self, tts, input_tt, care_set_tts, seq_enc, use_controllability_dont_care, tts_compressed=None, ffw = None):
        env = LogicNetworkEnv(tts,
                              self.num_inputs,
                              init_care_set_tt=care_set_tts,
                              ffw=ffw,
                              input_tt=input_tt,
                              max_length=self.max_seq_length,
                              use_controllability_dont_cares=use_controllability_dont_care,
                              tts_compressed=tts_compressed,
                              and_always_available=True)
        action_masks = []
        for token in seq_enc[:self.max_seq_length]:
            action_mask = env.action_masks[-1] # dec_env.gen_action_mask()
            assert action_mask[token]
            action_masks.append(action_mask)
            env.step(token)
        if len(seq_enc) < self.max_seq_length:
            action_masks.append(env.action_masks[-1])
        action_masks = np.stack(action_masks)
        if len(action_masks) < self.max_seq_length:
            action_masks_padding = np.zeros((self.max_seq_length - len(action_masks), self.vocab_size),
                                            dtype=bool)
            action_masks_padding[:, 0] = True
            action_masks = np.concatenate([action_masks, action_masks_padding], axis=0)
        return action_masks

    def load_and_encode(self, filename):
        with open(filename, 'r') as f:
            roots_aiger, num_ands, care_set_tts_str, opt_roots_aiger, opt_num_ands = json.load(f)
        roots, info = read_aiger(aiger_str=roots_aiger)
        opt_roots, _ = read_aiger(aiger_str=opt_roots_aiger)
        num_inputs, num_outputs = info[1], info[3]

        phase = np.random.rand(num_inputs) < 0.5
        perm = np.random.permutation(num_inputs)
        output_invs = np.random.rand(num_outputs) < 0.5
        roots = npn_transform(roots, phase, perm, output_invs)
        opt_roots = npn_transform(opt_roots, phase, perm, output_invs)

        if not isinstance(care_set_tts_str, list):
            care_set_tts = npn_transform_tt(bitarray.bitarray(care_set_tts_str), phase, perm, False)
        else:
            care_set_tts = [npn_transform_tt(bitarray.bitarray(care_set_tt_str), phase, perm, False)
                            for care_set_tt_str in care_set_tts_str]

        seq_enc, pos_enc = self._encode_postprocess(*encode_aig(roots, num_inputs))
        opt_seq_enc, opt_pos_enc = encode_aig(opt_roots, num_inputs)
        tts = compute_tts(roots, input_tt=self.input_tt)
        enc_action_masks = self.generate_action_masks(tts, self.input_tt, care_set_tts, seq_enc, True)
        dec_action_masks = self.generate_action_masks(tts, self.input_tt, care_set_tts, opt_seq_enc, True)
        opt_seq_enc, opt_pos_enc = self._encode_postprocess(opt_seq_enc, opt_pos_enc)
        return seq_enc, pos_enc, opt_seq_enc, opt_pos_enc, enc_action_masks, dec_action_masks

    def load_and_encode_formatted(self, filename):
        seq_enc, pos_enc, opt_seq_enc, opt_pos_enc, enc_action_mask, dec_action_mask = self.load_and_encode(filename)
        inputs = {
            'inputs': seq_enc,
            'enc_pos_encoding': pos_enc,
            'targets': opt_seq_enc,
            'dec_pos_encoding': opt_pos_enc,
            'enc_action_mask': enc_action_mask,
            'dec_action_mask': dec_action_mask
        }
        return inputs, opt_seq_enc

    def train(self,
              train_data_dir,
              ckpt_save_path=None,
              validation_split=0.1,
              epochs=10,
              initial_epoch=0,
              batch_size=128,
              profile=False,
              distributed=False,
              latest_ckpt_only=False,
              log_dir='tensorboard',
              excluded_files: list = None
              ):
        train_data_dir = train_data_dir + ("/" if train_data_dir[-1] != "/" else "")
        if ckpt_save_path is None:
            print("WARNING: ckpt_save_path is not specified, the trained model will not be saved during training!")
        else:
            ckpt_save_path = ckpt_save_path + ("/" if ckpt_save_path[-1] != "/" else "")

            if not os.path.exists(ckpt_save_path):
                os.mkdir(ckpt_save_path)

        train_files = os.listdir(train_data_dir)
        print("%d training files listed" % len(train_files))
        train_files.sort()
        print("training files sorted")
        np.random.seed(0)
        np.random.shuffle(train_files)
        print("training files shuffled")

        self._transformer.return_cache = False

        if excluded_files is not None:
            print("excluded files is not None, filtering training files...")
            excluded_files = set(excluded_files)
            new_train_files = []
            for file in train_files:
                if file not in excluded_files:
                    new_train_files.append(file)
            print("training files filtered, from %d to %d" % (len(train_files), len(new_train_files)))
            train_files = new_train_files

        train_files = [(train_data_dir + file) for file in train_files]
        self_copied = copy.copy(self)
        self_copied._transformer = None
        self_copied._transformer_inference = None

        mp_dataset = MPDataset(train_files, self_copied.load_and_encode_formatted, validation_split=validation_split, num_processes=8)

        output_signature = (
            {
                'inputs': tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
                'enc_pos_encoding': tf.TensorSpec(shape=(self.max_seq_length, self.max_tree_depth * 2),
                                                  dtype=tf.float32),
                'targets': tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
                'dec_pos_encoding': tf.TensorSpec(shape=(self.max_seq_length, self.max_tree_depth * 2),
                                                  dtype=tf.float32),
                'enc_action_mask': tf.TensorSpec(shape=(self.max_seq_length, self.vocab_size),
                                                 dtype=tf.bool),
                'dec_action_mask': tf.TensorSpec(shape=(self.max_seq_length, self.vocab_size),
                                                  dtype=tf.bool)
            }, tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32)
        )

        train_dataset = tf.data.Dataset.from_generator(mp_dataset.train_generator,
                                                       output_signature=output_signature) \
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset = tf.data.Dataset.from_generator(mp_dataset.validation_generator,
                                                            output_signature=output_signature) \
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)

        if profile:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            summary_writer = tf.summary.create_file_writer(log_dir)
            tf.summary.trace_on(profiler=True)

        if distributed:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                transformer = self._get_tf_transformer()
                if self.ckpt_path is not None:
                    transformer.load_weights(self.ckpt_path)
                learning_rate = CustomSchedule(self.embedding_width)
                optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
                transformer.compile(
                    optimizer=optimizer,
                    loss=masked_loss,
                    metrics=[masked_accuracy]
                )
        else:
            transformer = self._transformer
            learning_rate = CustomSchedule(self.embedding_width)
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            transformer.compile(
                optimizer=optimizer,
                loss=masked_loss,
                metrics=[masked_accuracy],
            )

        class LogCallback(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                # print(logs)
                pass

            def on_epoch_end(self, epoch, logs=None):
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                for stat in top_stats:
                    print(stat)


        log = LogCallback()

        callbacks = []
        if ckpt_save_path is not None:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_save_path + 'model-{epoch:04d}',
                save_weights_only=True,
                save_freq=(len(mp_dataset) * (epochs - initial_epoch) // batch_size) if latest_ckpt_only else 'epoch')
            callbacks.append(checkpoint)

        transformer.fit(train_dataset,
                        initial_epoch=initial_epoch,
                        epochs=epochs,
                        validation_data=validation_dataset,
                        callbacks=callbacks)
        mp_dataset.process.terminate()
        print("training finished")

        if profile:
            with summary_writer.as_default():
                tf.summary.trace_export(name="model_trace", profiler_outdir=log_dir)

        self._transformer.return_cache = True

    def optimize(self,
                 aigs: list,
                 context_num_inputs=None,
                 input_tts: list = None,
                 care_set_tts=None,
                 ffws=None,
                 context_hash_list=None,
                 num_mcts_steps=0,
                 num_leaf_parallelization=8,
                 num_mcts_playouts_per_step=10,
                 max_inference_seq_length=None,
                 max_inference_reward_list=None,
                 max_mcts_inference_seq_length=None,
                 use_controllability_dont_cares=True,
                 tts_compressed_list=None,
                 overflow_option='origin',
                 return_envs=False,
                 ):
        if self.ckpt_path is None:
            print("no checkpoint loaded, downloading from https://huggingface.co/snowkylin/circuit-transformer ...")
            self.load_from_hf()

        if max_inference_seq_length is None:
            max_inference_seq_length = self.max_seq_length
        if max_mcts_inference_seq_length is None:
            max_mcts_inference_seq_length = max_inference_seq_length

        optimized_aigs = []
        for i in range(0, len(aigs), self.inference_batch_size):
            aigs_batch = aigs[i: i + self.inference_batch_size]
            care_set_tts_batch = care_set_tts[i: i + self.inference_batch_size] if care_set_tts is not None else None
            ffws_batch = ffws[i: i + self.inference_batch_size] if ffws is not None else None
            input_tts_batch = input_tts[i: i + self.inference_batch_size] if input_tts is not None else None
            context_hash_list_batch = context_hash_list[i: i + self.inference_batch_size] if context_hash_list is not None else None
            tts_compressed_batch = tts_compressed_list[i: i + self.inference_batch_size] if tts_compressed_list is not None else None
            max_inference_reward_list_batch = max_inference_reward_list[i: i + self.inference_batch_size] if max_inference_reward_list is not None else None
            optimized_aigs += self.optimize_batch(aigs_batch,
                                                  max_inference_seq_length,
                                                  max_mcts_inference_seq_length,
                                                  context_num_inputs,
                                                  input_tts_batch,
                                                  care_set_tts_batch,
                                                  ffws_batch,
                                                  context_hash_list_batch,
                                                  max_inference_reward_list_batch,
                                                  num_mcts_steps,
                                                  num_leaf_parallelization,
                                                  num_mcts_playouts_per_step,
                                                  use_controllability_dont_cares,
                                                  tts_compressed_batch,
                                                  overflow_option,
                                                  return_envs)
        return optimized_aigs

    def optimize_batch(self,
                       aigs: list,
                       max_inference_seq_length,
                       max_mcts_inference_seq_length=None,
                       context_num_inputs=None,
                       input_tts: list = None,
                       care_set_tts=None,
                       ffws=None,
                       context_hash_list=None,
                       max_inference_reward_list=None,
                       num_mcts_steps=0,
                       num_leaf_parallelization=8,
                       num_mcts_playouts_per_step=10,
                       use_controllability_dont_cares=True,
                       tts_compressed=None,
                       overflow_option='origin',
                       return_envs=False,
                       return_mcts_roots=False,
                       return_input_encodings=False,
                       puct_explore_ratio=1.
                       ):
        total_time = time.time()
        start_time = time.time()
        encoded_aigs = []
        aigs = aigs.copy()
        if max_mcts_inference_seq_length is None:
            max_mcts_inference_seq_length = max_inference_seq_length
        tts_list = []
        enc_action_masks = []
        orig_aig_size = []
        for i, aig in enumerate(aigs):
            if isinstance(aig, str):
                aigs[i], info = read_aiger(aig)
            if len(aigs[i]) > 2:
                raise OverflowError("the number of outputs for input aig network should be <= 2 "
                                    "as the default model is trained on 8-input, 2-output networks")
            orig_aig_size.append(count_num_ands(aigs[i]))
            seq_enc, pos_enc = encode_aig(aigs[i], self.num_inputs)
            input_tt = self.input_tt if input_tts is None else input_tts[i]
            tts = [compute_tt(root, input_tt=input_tt) for root in aig]
            enc_action_masks.append(self.generate_action_masks(tts,
                                                               input_tt,
                                                               None if care_set_tts is None else care_set_tts[i],
                                                               seq_enc,
                                                               use_controllability_dont_care=use_controllability_dont_cares,
                                                               tts_compressed=None if tts_compressed is None else tts_compressed[i]))
            encoded_aigs.append(self._encode_postprocess(seq_enc, pos_enc))
            tts_list.append(tts)
        enc_action_masks = np.stack(enc_action_masks)
        seq_enc, pos_enc = tuple(map(lambda x: np.stack(x, axis=0), zip(*encoded_aigs)))
        batch_size = len(aigs)

        inputs = {'inputs': seq_enc, 'enc_pos_encoding': pos_enc, 'enc_action_mask': enc_action_masks}
        targets = np.zeros((batch_size, 1), dtype=np.int32)
        dec_pos_encoding = np.zeros((batch_size, 1, self.max_tree_depth * 2), dtype=np.float32)
        envs = [LogicNetworkEnv(
            tts=tts_list[i],
            num_inputs=self.num_inputs,
            context_num_inputs=context_num_inputs,
            input_tt=self.input_tt if input_tts is None else input_tts[i],
            init_care_set_tt=None if care_set_tts is None else care_set_tts[i],
            ffw=None if ffws is None else ffws[i],
            context_hash=None if context_hash_list is None else context_hash_list[i],
            max_tree_depth=self.max_tree_depth,
            max_length=max_inference_seq_length,
            max_inference_reward=None if max_inference_reward_list is None else max_inference_reward_list[i],
            use_controllability_dont_cares=use_controllability_dont_cares,
            tts_compressed=None if tts_compressed is None else tts_compressed[i],
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            verbose=self.verbose)
            for i, aig in enumerate(aigs)]
        init_mcts_roots = [MCTSNode(None, 0, None, info={'env': env, 'reward': None, 'done': None, 'rollout_success': None}, puct_explore_ratio=puct_explore_ratio) for env in self._copy_env(envs)]
        transformer_time = 0.
        action_mask_time = 0.
        step_time = 0.
        init_time = time.time() - start_time
        if self.verbose > 0:
            print("optimization initialized, time cost %.2f" % init_time)
        for i in range(max_inference_seq_length):
            if all([e.is_finished for e in envs]):
                break
            inputs['targets'], inputs['dec_pos_encoding'] = targets, dec_pos_encoding

            # generate action mask
            start_time = time.time()
            action_masks = np.stack([e.action_masks[i] for e in envs], axis=0)
            inputs['dec_action_mask'] = np.expand_dims(action_masks, axis=1)
            action_mask_time += time.time() - start_time

            start_time = time.time()
            policy, cache = self._transformer_inference(inputs, return_kv_cache=True, return_last_token=True)
            transformer_time += time.time() - start_time
            inputs['cache'] = cache

            start_time = time.time()
            tokens = np.argmax(policy, axis=1)

            if num_mcts_steps > 0:
                if i < num_mcts_steps:
                    best_token_seqs, mcts_roots = self._batch_MCTS_policy_with_leaf_parallelization(envs,
                                                                          max_inference_seq_length=max_mcts_inference_seq_length,
                                                                          num_leaf_parallelizations=num_leaf_parallelization,
                                                                          num_playouts=num_mcts_playouts_per_step,
                                                                          src_tokens=seq_enc,
                                                                          src_pos_enc=pos_enc,
                                                                          src_action_mask=enc_action_masks,
                                                                          roots=init_mcts_roots if i == 0 else mcts_roots,
                                                                          orig_aigs_size=orig_aig_size,
                                                                          puct_explore_ratio=puct_explore_ratio)
                    tokens = [b[0] for b in best_token_seqs]
                else:
                    if i == num_mcts_steps:
                        if self.verbose > 1:
                            print("best_token_seqs", best_token_seqs)
                    for j, b in enumerate(best_token_seqs):
                        if len(b) >= i - num_mcts_steps + 2:
                            tokens[j] = b[i - num_mcts_steps + 1]

            rewards, dones = zip(*[e.step(token) for token, e in zip(tokens, envs)])
            pos_encodings = [e.positional_encodings[-1] for e in envs]

            pos_encodings = np.expand_dims(np.stack(pos_encodings, axis=0),
                                           axis=1)  # [batch_size, 1, max_tree_depth * 2]
            targets_new = np.expand_dims(tokens, axis=1)
            if self.use_kv_cache:
                targets = targets_new
                dec_pos_encoding = pos_encodings
            else:
                targets = np.concatenate([targets, targets_new], axis=1)
                dec_pos_encoding = np.concatenate([dec_pos_encoding, pos_encodings], axis=1)
            step_time += time.time() - start_time
            if self.verbose > 0:
                print(i, tokens)
        if return_envs and self.verbose == 0:
            return envs
        optimized_aigs = []
        num_succeed_aigs = 0
        total_gain, seq_total_gain_for_succeeded_aig, seq_total_gain = 0, 0, 0
        for i, (aig, env) in enumerate(zip(aigs, envs)):
            orig_num_ands = count_num_ands(aig)
            if self.verbose > 1:
                seq_roots, info = sequential_synthesis(aig)
                seq_total_gain += orig_num_ands - info[4]
            assert env.success == (len(env.roots) == len(aig) and check_integrity(env.roots))
            if env.success:
                num_succeed_aigs += 1
                if not return_envs:
                    optimized_aigs.append(env.roots)
                if self.verbose > 0:
                    num_ands = count_num_ands(env.roots)
                    total_gain += max(orig_num_ands - num_ands, 0)
                    print("aig #%d successfully optimized, #(AND) from %d to %d, cumulative reward %d, gain = %d" %
                          (i, orig_num_ands, num_ands, env.cumulative_reward, orig_num_ands - num_ands),
                          end="" if self.verbose > 1 else "\n")
                    if self.verbose > 1:
                        seq_total_gain_for_succeeded_aig += orig_num_ands - info[4]
                        print(" (resyn2: %d)" % info[4])
            else:
                if self.verbose > 0:
                    print("aig #%d (#(AND) = %d) failed to be optimized%s" %
                          (i, orig_num_ands, ', use original aig instead' if overflow_option == 'origin' else ''))
                    if self.verbose > 1:
                        print(" (resyn2: %d)" % info[4])
                if not return_envs:
                    optimized_aigs.append(aig if overflow_option == 'origin' else env.roots)
        if self.verbose > 0:
            print(
                "%d out of %d aigs successfully optimized, total time %.2f, init time %.2f, transformer time %.2f, action mask time %.2f, step time %.2f" %
                (num_succeed_aigs, len(aigs), time.time() - total_time, init_time, transformer_time, action_mask_time, step_time))
            if num_succeed_aigs > 0:
                print("average gain %.3f for successfully optimized aigs" % (total_gain / num_succeed_aigs))
            print("average gain %.3f for all aigs (failed aigs correspond to zero gain)" % (total_gain / len(aigs)))
            if self.verbose > 1:
                if num_succeed_aigs > 0:
                    print("resyn2: %.3f / %.3f" % (
                        seq_total_gain_for_succeeded_aig / num_succeed_aigs, seq_total_gain / len(aigs)))
        if not return_envs and not return_mcts_roots:
            return optimized_aigs
        ret = []
        if return_envs:
            ret.append(envs)
        if return_mcts_roots:
            ret.append(init_mcts_roots)
        if return_input_encodings:
            ret.append({'inputs': seq_enc, 'enc_pos_encoding': pos_enc, 'enc_action_mask': enc_action_masks})
        return tuple(ret) if len(ret) > 1 else ret[0]


if __name__ == "__main__":
    circuit_transformer = CircuitTransformer()
    aig0, info0 = read_aiger(aiger_str="""aag 33 8 0 2 25
2\n4\n6\n8\n10\n12\n14\n16\n58\n67
18 13 16\n20 19 7\n22 21 15\n24 3 9\n26 25 11
28 27 17\n30 3 6\n32 29 31\n34 29 32\n36 23 35
38 7 36\n40 10 29\n42 41 32\n44 13 15\n46 42 45
48 47 21\n50 39 49\n52 4 45\n54 25 53\n56 54 5
58 51 57\n60 45 12\n62 18 61\n64 63 19\n66 48 64
""")
    aig1, info1 = read_aiger(aiger_str="""aag 22 8 0 2 14
2\n4\n6\n8\n10\n12\n14\n16\n24\n44
18 10 12\n20 8 7\n22 21 5\n24 19 23\n26 11 3
28 6 4\n30 26 28\n32 8 5\n34 32 26\n36 35 17
38 37 7\n40 31 39\n42 41 12\n44 43 15
""")
    aigs = [aig0, aig1]

    optimized_aigs = circuit_transformer.optimize(aigs)
    print("Circuit Transformer:")
    for i, (aig, optimized_aig) in enumerate(zip(aigs, optimized_aigs)):
        print("aig %d #(AND) from %d to %d, equivalence check: %r" %
              (i, count_num_ands(aig), count_num_ands(optimized_aig), cec(aig, optimized_aig)))

    optimized_aigs_with_mcts = circuit_transformer.optimize(
        aigs=aigs,
        num_mcts_steps=1,
        num_mcts_playouts_per_step=10
    )
    print("Circuit Transformer + Monte-Carlo Tree Search:")
    for i, (aig, optimized_aig) in enumerate(zip(aigs, optimized_aigs_with_mcts)):
        print("aig %d #(AND) from %d to %d, equivalence check: %r" %
              (i, count_num_ands(aig), count_num_ands(optimized_aig), cec(aig, optimized_aig)))
