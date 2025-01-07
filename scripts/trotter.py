import argparse
import math
import qiskit
import qiskit.qasm2
import random
import sys

from enum import auto, Enum


class Axis(Enum):
    X = auto(),
    ZZ = auto(),


class ExpOperator():
    def __init__(self, axis: Axis, indices: list[int], exponent: float):
        self.axis = axis
        self.indices = indices
        self.exponent = exponent

    def commute_with(self, other: 'ExpOperator') -> bool:
        return self.axis == other.axis

    def mult(self, other: 'ExpOperator') -> 'ExpOperator':
        assert self.axis == other.axis
        assert self.indices == other.indices
        return ExpOperator(self.axis, self.indices, self.exponent + other.exponent)


def u2(delta: float, ops: list[ExpOperator], j: float, g: float, width: int) -> None:
    for i in range(width):
        for k in range(width):
            ops.append(ExpOperator(Axis.X, [i * width + k], g * -delta / 2))

    for i in range(width):
        for k in range(width):
            if i < width - 1:
                indices = [i * width + k, (i + 1) * width + k]
                ops.append(ExpOperator(Axis.ZZ, indices, -j * -delta / 2))
            if k < width - 1:
                indices = [i * width + k, i * width + k + 1]
                ops.append(ExpOperator(Axis.ZZ, indices, -j * -delta / 2))

    for i in range(width):
        for k in range(width):
            ops.append(ExpOperator(Axis.X, [i * width + k], g * -delta / 2))


def u4(delta: float, ops: list[ExpOperator], j: float, g: float, width: int) -> None:
    gamma = 1 / (4 - 4 ** (1 / 3))
    u2(gamma * delta, ops, j, g, width)
    u2(gamma * delta, ops, j, g, width)
    u2((1 - 4 * gamma) * delta, ops, j, g, width)
    u2(gamma * delta, ops, j, g, width)
    u2(gamma * delta, ops, j, g, width)


def main() -> None:
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--width', type=int, default=6)
    parser.add_argument('--J', type=float, help='the coefficient of the ZZ terms')
    parser.add_argument('--g', type=float, help='the coefficient of the X terms')
    parser.add_argument('--t', type=float, help='the total time')
    parser.add_argument('--T', type=int, help='the number of Trotter steps')

    args = parser.parse_args()

    ops: list[ExpOperator] = []
    for _ in range(args.T):
        u4(args.t / args.T, ops, args.J, args.g, args.width)
    id_operator_indices: set[int] = set()
    for i in range(len(ops)):
        if i in id_operator_indices:
            continue
        for j in range(i + 1, len(ops)):
            if j in id_operator_indices:
                continue
            if not ops[i].commute_with(ops[j]):
                break
            if ops[i].indices == ops[j].indices:
                ops[i] = ops[i].mult(ops[j])
                id_operator_indices.add(j)
    ops = [op for (i, op) in enumerate(ops) if i not in id_operator_indices]

    circuit = qiskit.QuantumCircuit(args.width * args.width)

    for op in ops:
        match op.axis:
            case Axis.X:
                assert len(op.indices) == 1
                circuit.h(op.indices[0])
                circuit.rz(-op.exponent * 2, op.indices[0])
                circuit.h(op.indices[0])
            case Axis.ZZ:
                assert len(op.indices) == 2
                circuit.cx(op.indices[0], op.indices[1])
                circuit.rz(-op.exponent * 2, op.indices[1])
                circuit.cx(op.indices[0], op.indices[1])

    qiskit.qasm2.dump(circuit, sys.stdout)


if __name__ == '__main__':
    main()
