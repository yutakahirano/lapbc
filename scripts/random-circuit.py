import math
import qiskit
import random
import sys

# https://arxiv.org/abs/1608.00263
def random_circuit_round(qc):
    def qubit_index(c, r):
        return 6 * r + c

    two_qubit_gates_in_layers = [[
        # Layer1
        ((2, 0), (3, 0)),
        ((0, 1), (1, 1)), ((4, 1), (5, 1)),
        ((2, 2), (3, 2)),
        ((0, 3), (1, 3)), ((4, 3), (5, 3)),
        ((2, 4), (3, 4)),
        ((0, 5), (1, 5)), ((4, 5), (5, 5)),
    ], [
        # Layer2
        ((0, 0), (1, 0)), ((4, 0), (5, 0)),
        ((2, 1), (3, 1)),
        ((0, 2), (1, 2)), ((4, 2), (5, 2)),
        ((2, 3), (3, 3)),
        ((0, 4), (1, 4)), ((4, 4), (5, 4)),
        ((2, 5), (3, 5)),
    ], [
        # Layer3
        ((1, 1), (1, 2)), ((3, 1), (3, 2)), ((5, 1), (5, 2)),
        ((0, 3), (0, 4)), ((2, 3), (2, 4)), ((4, 3), (4, 4)),
    ], [
        # Layer4
        ((0, 1), (0, 2)), ((2, 1), (2, 2)), ((4, 1), (4, 2)),
        ((1, 3), (1, 4)), ((3, 3), (3, 4)), ((5, 3), (5, 4)),
    ], [
        # Layer5
        ((3, 0), (4, 0)),
        ((1, 1), (2, 1)),
        ((3, 2), (4, 2)),
        ((1, 3), (2, 3)),
        ((3, 4), (4, 4)),
        ((1, 5), (2, 5)),
    ], [
        # Layer6
        ((1, 0), (2, 0)),
        ((3, 1), (4, 1)),
        ((1, 2), (2, 2)),
        ((3, 3), (4, 3)),
        ((1, 4), (2, 4)),
        ((3, 5), (4, 5)),
    ], [
        # Layer7
        ((0, 0), (0, 1)), ((2, 0), (2, 1)), ((4, 0), (4, 1)),
        ((1, 2), (1, 3)), ((3, 2), (3, 3)), ((5, 2), (5, 3)),
        ((0, 4), (0, 5)), ((2, 4), (2, 5)), ((4, 4), (4, 5)),
    ], [
        # Layer8
        ((1, 0), (1, 1)), ((3, 0), (3, 1)), ((5, 0), (5, 1)),
        ((0, 2), (0, 3)), ((2, 2), (2, 3)), ((4, 2), (4, 3)),
        ((1, 4), (1, 5)), ((3, 4), (3, 5)), ((5, 4), (5, 5)),
    ]]

    all_qubits = set(range(36))
    t_count_dict = {q: 0 for q in all_qubits}
    for two_qubit_gates in two_qubit_gates_in_layers:
        free_qubits = set(all_qubits)

        for ((c1, r1), (c2, r2)) in two_qubit_gates:
            q1 = qubit_index(c1, r1)
            q2 = qubit_index(c2, r2)

            assert q1 != q2
            assert q1 in free_qubits
            assert q2 in free_qubits

            free_qubits.remove(q1)
            free_qubits.remove(q2)

            qc.h(q2)
            qc.cx(q1, q2)
            qc.h(q2)

        for q in free_qubits:
            match random.choice(['sx', 'sy', 't']):
                case 'sx':
                    if t_count_dict[q] > 0:
                        qc.rz(t_count_dict[q] * math.pi / 4, q)
                        t_count_dict[q] = 0
                    qc.sx(q)
                case 'sy':
                    if t_count_dict[q] > 0:
                        qc.rz(t_count_dict[q] * math.pi / 4, q)
                        t_count_dict[q] = 0
                    qc.ry(math.pi / 2, q)
                case 't':
                    t_count_dict[q] += 1
                case _:
                    assert False  
    for q in all_qubits:
        if t_count_dict[q] > 0:
            qc.rz(t_count_dict[q] * math.pi / 4, q)
            t_count_dict[q] = 0


def main():
    qc = qiskit.QuantumCircuit(36)
    for i in range(36):
        qc.h(i)
    
    num_rounds = 8
    for i in range(num_rounds):
        random_circuit_round(qc)

    qiskit.qasm2.dump(qc, sys.stdout)

if __name__ == "__main__":
    main()
