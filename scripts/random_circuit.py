import argparse
import math
import qiskit
import qiskit.qasm2
import random
import sys


def generate_two_qubit_gates_in_layers(width: int) -> list[list[tuple[tuple[int, int], tuple[int, int]]]]:
    assert width % 2 == 0

    two_qubit_gates_in_layers: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []

    def append(p1: tuple[int, int], p2: tuple[int, int]) -> None:
        (x1, y1) = p1
        (x2, y2) = p2
        if x1 < 0 or x1 >= width or y1 < 0 or y1 >= width or x2 < 0 or x2 >= width or y2 < 0 or y2 >= width:
            return
        two_qubit_gates_in_layers[-1].append((p1, p2))

    two_qubit_gates_in_layers.append([])
    for y in range(0, width, 2):
        for x in range(2, width, 4):
            append((x, y), (x + 1, y))
        for x in range(0, width, 4):
            append((x, y + 1), (x + 1, y + 1))

    two_qubit_gates_in_layers.append([])
    for y in range(0, width, 2):
        for x in range(0, width, 4):
            append((x, y), (x + 1, y))
        for x in range(2, width, 4):
            append((x, y + 1), (x + 1, y + 1))

    two_qubit_gates_in_layers.append([])
    for y in range(1, width, 4):
        for x in range(1, width, 2):
            append((x, y), (x, y + 1))
    for y in range(3, width, 4):
        for x in range(0, width, 2):
            append((x, y), (x, y + 1))

    two_qubit_gates_in_layers.append([])
    for y in range(1, width, 4):
        for x in range(0, width, 2):
            append((x, y), (x, y + 1))
    for y in range(3, width, 4):
        for x in range(1, width, 2):
            append((x, y), (x, y + 1))

    two_qubit_gates_in_layers.append([])
    for y in range(0, width, 2):
        for x in range(3, width, 4):
            append((x, y), (x + 1, y))
        for x in range(1, width, 4):
            append((x, y + 1), (x + 1, y + 1))

    two_qubit_gates_in_layers.append([])
    for y in range(0, width, 2):
        for x in range(1, width, 4):
            append((x, y), (x + 1, y))
        for x in range(3, width, 4):
            append((x, y + 1), (x + 1, y + 1))

    two_qubit_gates_in_layers.append([])
    for y in range(0, width, 4):
        for x in range(0, width, 2):
            append((x, y), (x, y + 1))
        for x in range(1, width, 2):
            append((x, y + 2), (x, y + 3))

    two_qubit_gates_in_layers.append([])
    for y in range(0, width, 4):
        for x in range(1, width, 2):
            append((x, y), (x, y + 1))
        for x in range(0, width, 2):
            append((x, y + 2), (x, y + 3))

    return two_qubit_gates_in_layers


# https://arxiv.org/abs/1608.00263
def random_circuit(qc: qiskit.QuantumCircuit, width: int, num_rounds: int) -> None:
    def qubit_index(c, r):
        return width * r + c

    def flush_t_gates(q: int):
        if t_count_dict[q] > 0:
            qc.rz(t_count_dict[q] * math.pi / 4, q)
            t_count_dict[q] = 0

    two_qubit_gates_in_layers = generate_two_qubit_gates_in_layers(width)

    all_qubits = set(range(width * width))
    t_count_dict = {q: 0 for q in all_qubits}
    for _ in range(num_rounds):
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

                # We don't need to flush T gates here, because T and CZ commute.

                qc.h(q2)
                qc.cx(q1, q2)
                qc.h(q2)

            for q in free_qubits:
                match random.choice(['sx', 'sy', 't']):
                    case 'sx':
                        flush_t_gates(q)
                        qc.sx(q)
                    case 'sy':
                        flush_t_gates(q)
                        qc.ry(math.pi / 2, q)
                    case 't':
                        t_count_dict[q] += 1
                    case _:
                        assert False
    for q in all_qubits:
        flush_t_gates(q)


def main():
    parser = argparse.ArgumentParser(
        description='description')
    parser.add_argument('--width', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args()

    if args.width < 5:
        raise ValueError('width must be at least 5')
    if args.width % 2 != 0:
        raise ValueError('width must be even')
    if args.rounds < 1:
        raise ValueError('rounds must be at least 1')
    
    qc = qiskit.QuantumCircuit(args.width * args.width)
    for i in range(args.width * args.width):
        qc.h(i)

    random_circuit(qc, args.width, args.rounds)

    qiskit.qasm2.dump(qc, sys.stdout)


if __name__ == "__main__":
    main()
