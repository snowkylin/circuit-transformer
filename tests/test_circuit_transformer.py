import bitarray


def test_circuit_transformer():
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
    optimized_aigs = circuit_transformer.optimize(aigs)  # batch operation that accepts a list of aigs
    optimized_aig = optimized_aigs[0]

    print("Optimized AIG: \n%s\n#(AND) from %d to %d, equivalence check: %r" %
          (ct.write_aiger(optimized_aig), ct.count_num_ands(aig),
           ct.count_num_ands(optimized_aig), ct.cec(aig, optimized_aig)))

    # Monte-Carlo tree search (MCTS) can be applied to further boost
    # the performance of size minimization (with additional time cost)
    optimized_aigs_with_mcts = circuit_transformer.optimize(
        aigs=aigs,
        num_mcts_steps=1,               # enable MCTS
        num_mcts_playouts_per_step=2   # number of MCTS playouts
    )
    optimized_aig_with_mcts = optimized_aigs_with_mcts[0]
    print("Optimized AIG: \n%s\n#(AND) from %d to %d, equivalence check: %r" %
          (ct.write_aiger(optimized_aig_with_mcts), ct.count_num_ands(aig),
           ct.count_num_ands(optimized_aig_with_mcts), ct.cec(aig, optimized_aig)))


def test_aig():
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
    tts = ct.compute_tts(aig, num_inputs=2)

    assert ct.count_num_ands(aig) == 2
    assert tts[0] == bitarray.bitarray('0010')
    assert tts[1] == bitarray.bitarray('1011')


if __name__ == "__main__":
    test_circuit_transformer()
    test_aig()