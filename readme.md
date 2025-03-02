# Circuit Transformer

This is the official implementation of the Circuit Transformer model in the ICLR 2025 paper "[Circuit Transformer: A Transformer That Preserves Logical Equivalence](https://openreview.net/forum?id=kpnW12Lm9p)"

The model checkpoints are in a separate repository in HugggingFace: <https://huggingface.co/snowkylin/circuit-transformer>

## Install

Circuit Transformer is available on PyPI as `circuit-transformer`.

```bash
pip install circuit-transformer
```

Circuit Transformer use [TensorFlow](https://www.tensorflow.org) as the backend. GPU acceleration is highly recommended for significantly faster inference, which requires

- An NVIDIA GPU with CUDA support
- Linux environment (for Windows, you can use WSL2)

More details can be found [here](https://www.tensorflow.org/install/pip). To verify the GPU status, please run

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You need to install GraphViz separately as follows if you need to visualize the network.

```bash
conda install graphviz
```

## Usage

An interactive Colab demo can be found [[here]](https://colab.research.google.com/drive/1r0tS_ZbxUf0ojkazViT94ENMBq4r7B_t?usp=sharing).

The behavior of Circuit Transformer depends on the data it is trained on. With a default checkpoint, the Circuit Transformer conducts size minimization of [And-Inverter Graphs (AIGs)](https://fmv.jku.at/aiger/) with number of inputs <= 8 and number of outputs <= 2, since it is trained on 8-input, 2-output AIG pairs `(original AIG, size-minimized AIG)` in a supervised way. Empirically, the default checkpoint works well for AIGs within 30 AND nodes.

As a Transformer model, Circuit Transformer is more efficient in batch mode. That is, optimizing multiple AIGs simultaneously (with a single API call) will be much faster than optimizing them one-by-one. Therefore, `CircuitTransformer.optimize` accepts a list of AIGs by default, and return a list of optimized AIGs. Empirically, the model runs efficiently (with higher GPU utilization) if the number of AIGs is larger than 128.

```python
import circuit_transformer as ct

# load a AIG logic network with Aiger format
# see https://fmv.jku.at/aiger/ for details
aig, info = ct.read_aiger(aiger_str="""aag 33 8 0 2 25
2\n4\n6\n8\n10\n12\n14\n16\n58\n67
18 13 16\n20 19 7\n22 21 15\n24 3 9\n26 25 11
28 27 17\n30 3 6\n32 29 31\n34 29 32\n36 23 35
38 7 36\n40 10 29\n42 41 32\n44 13 15\n46 42 45
48 47 21\n50 39 49\n52 4 45\n54 25 53\n56 54 5
58 51 57\n60 45 12\n62 18 61\n64 63 19\n66 48 64
""")

circuit_transformer = ct.CircuitTransformer()
aigs = [aig]    # in practice, it is encouraged to optimize multiple AIGs in a single API call
optimized_aigs = circuit_transformer.optimize(aigs) # batch operation that accepts a list of aigs
optimized_aig = optimized_aigs[0]

print("Optimized AIG: \n%s\n#(AND) from %d to %d, equivalence check: %r" %
      (ct.write_aiger(optimized_aig), 
       ct.count_num_ands(aig), 
       ct.count_num_ands(optimized_aig), 
       ct.cec(aig, optimized_aig)))
```

The output should be (some newlines are replaced by `\n`)

```
Optimized AIG: 
aag 24 8 0 2 16
2\n4\n6\n8\n10\n12\n14\n16\n44\n49
18 13 15\n20 19 11\n22 13 16\n24 23 7\n26 21 25
28 3 9\n30 3 6\n32 31 11\n34 29 32\n36 16 7
38 35 37\n40 27 39\n42 29 5\n44 41 43\n46 41 23
48 46 6

#(AND) from 25 to 16, equivalence check: True
```

### Monte-Carlo tree search

Monte-Carlo tree search (MCTS) can be applied to further boost the performance of size minimization (with additional time cost). The result should be comparable to `&deepsyn` in [ABC](https://people.eecs.berkeley.edu/~alanmi/abc/).

```python
optimized_aigs_with_mcts = circuit_transformer.optimize(
    aigs=aigs,
    num_mcts_steps=1,               # enable MCTS
    num_mcts_playouts_per_step=10   # number of MCTS playouts
)
optimized_aig_with_mcts = optimized_aigs_with_mcts[0]
print("Optimized AIG: \n%s\n#(AND) from %d to %d, equivalence check: %r" %
      (ct.write_aiger(optimized_aig_with_mcts), 
       ct.count_num_ands(aig),
       ct.count_num_ands(optimized_aig_with_mcts), 
       ct.cec(aig, optimized_aig)))
deepsyn_optimized_aig, info = ct.sequential_synthesis(aig, command='deepsyn')
print("deepsyn optimized AIG: #(AND) = %d" % ct.count_num_ands(deepsyn_optimized_aig))
```

The output should be

```
Optimized AIG: 
aag 22 8 0 2 14
2\n4\n6\n8\n10\n12\n14\n16\n42\n45
18 3 9\n20 19 5\n22 19 11\n24 23 17\n26 3 6
28 27 11\n30 13 15\n32 28 31\n34 13 16\n36 35 7
38 33 37\n40 25 39\n42 21 41\n44 38 35

#(AND) from 25 to 14, equivalence check: True
deepsyn optimized AIG: #(AND) = 14
```

### AIG representation in this repository

Sometimes you may need to manipulate the AIGs on your own. This repository represents AIG with two classes: `Node` and `NodeWithInv`.

- `Node(var, left, right)`: a 2-input AND node or a primary input
  - `var=None`: a 2-input AND node, for which the `left` and `right` parameters are two `NodeWithInv` instances denoting the 1st input and 2nd input of the node.
  - `var` is an non-negative integer: the `var`-th primary input
- `NodeWithInv(node, inverted)`: a wrapper including a `Node` instance and an `inverted` attribute, denoting whether the `Node` instance is inverted (`True` = inverted).

Then, an AIG is simply a list of all the output nodes.

For example, you can build a 2-input, 2-output AIG 
- `o_0 = ~x_0 & x_1`
- `o_1 = ~(x_0 & ~x_1)`

as follows

```python
import circuit_transformer as ct

x_0 = ct.Node(var=0, left=None, right=None)
x_1 = ct.Node(var=1, left=None, right=None)
and_0 = ct.Node(var=None, 
                left=ct.NodeWithInv(x_0, inverted=True), 
                right=ct.NodeWithInv(x_1, inverted=False))
and_1 = ct.Node(var=None, 
                left=ct.NodeWithInv(x_0, inverted=False), 
                right=ct.NodeWithInv(x_1, inverted=True))
aig = [ct.NodeWithInv(and_0, inverted=False), ct.NodeWithInv(and_1, inverted=True)]

print(ct.count_num_ands(aig))
print(ct.compute_tts(aig, num_inputs=2))
```

which will output the total number of nodes, and the truth tables of the AIG:
``` 
2
[bitarray('0010'), bitarray('1011')]
```

Some helper functions for AIG manipulation are as follows:
- `ct.read_aiger(filename=None, aiger_str=None)`: convert an Aiger file (`filename`) or an Aiger string (`aiger_str`) to the aforementioned AIG representation. return `(aig, info)` in which `aig` is a list of output nodes and `info` is the first five numbers of Aiger format.
- `ct.write_aiger(aig, filename)`: convert `aig` to Aiger format, return an Aiger string, and also write to `filename` if specified.
- `ct.count_num_ands(aig)`: return the number of AND nodes of `aig`
- `ct.compute_tts(aig, num_inputs)`: return the truth tables for each output of `aig` with `num_inputs` primary inputs (for the truth table for a single node, use `ct.compute_tt`)

There are also some helper functions with ABC as the backend, including
- `ct.sequential_synthesis(aig, command)`: optimize `aig` with ABC's `resyn2` (`command="resyn2"`) or `&deepsyn` (`command="deepsyn"`) and return the optimized AIG.
- `ct.cec(aig1, aig2)`: check whether `aig1` and `aig2` are equivalent with ABC's `cec` command, return `True` if equivalent.

### Don't Cares

Circuit Transformer naturally supports [don't cares](https://en.wikipedia.org/wiki/Don%27t-care_term). That is, only a subset of the `2^N` terms should be kept equivalent to the original circuit. To enable don't cares, feed the `care_set_tts` parameter of `CircuitTransformer.optimize` with a list of care sets corresponding to each input AIGs. A care set is a `bitarray.bitarray` of size `2^N`, in which the terms to be cared is equal to 1. For example, a care set of `bitarray.bitarray('0110')` for a 2-input AIG means the output circuit should be equivalent on `x_0=1,x_1=0` and `x_0=0,x_1=1`.

## Build

```bash
python setup.py sdist
```

To install it locally, run

```bash
pip install dist/circuit-transformer-x.x.tar.gz
```

To upload to PyPI, run

```bash
twine upload dist/circuit-transformer-x.x.tar.gz
```

## Citation

```
@inproceedings{li2025circuit,
    title={Circuit Transformer: A Transformer That Preserves Logical Equivalence},
    author={Xihan Li and Xing Li and Lei Chen and Xing Zhang and Mingxuan Yuan and Jun Wang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=kpnW12Lm9p}
}
```