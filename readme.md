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

```python
import circuit_transformer as ct

# load a logic network with And-Inverter Graph (AIG) format
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
# Circuit Transformer works with batch operation
# i.e., optimize multiple aigs simultaneously
aigs = [aig]
optimized_aigs = circuit_transformer.optimize(aigs) # batch operation that accepts a list of aigs
optimized_aig = optimized_aigs[0]

print("Optimized AIG: \n%s\n#(AND) from %d to %d, equivalence check: %r" %
      (ct.write_aiger(optimized_aig), ct.count_num_ands(aig), 
       ct.count_num_ands(optimized_aig), ct.cec(aig, optimized_aig)))
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

Monte-Carlo tree search (MCTS) can be applied to further boost the performance of size minimization (with additional time cost). The result should be comparable with `&deepsyn` in [ABC](https://people.eecs.berkeley.edu/~alanmi/abc/).

```python
optimized_aigs_with_mcts = circuit_transformer.optimize(
    aigs=aigs,
    num_mcts_steps=1,               # enable MCTS
    num_mcts_playouts_per_step=10   # number of MCTS playouts
)
optimized_aig_with_mcts = optimized_aigs_with_mcts[0]
print("Optimized AIG: \n%s\n#(AND) from %d to %d, equivalence check: %r" %
      (ct.write_aiger(optimized_aig_with_mcts), ct.count_num_ands(aig),
       ct.count_num_ands(optimized_aig_with_mcts), ct.cec(aig, optimized_aig)))
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