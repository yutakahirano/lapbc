use super::pbc::Axis;
use super::pbc::Operation;
use super::pbc::Pauli;
use super::pbc::PauliRotation;

// Representing a single-qubit Clifford that maps Pauli to Pauli through conjugation.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SingleQubitCliffordConjugation {
    from_x: Pauli,
    from_y: Pauli,
    from_z: Pauli,
    qubit_index: usize,
}

impl SingleQubitCliffordConjugation {
    fn new_empty(qubit_index: usize) -> Self {
        Self {
            from_x: Pauli::X,
            from_y: Pauli::Y,
            from_z: Pauli::Z,
            qubit_index,
        }
    }
    fn new(rotation: &PauliRotation) -> Self {
        assert!(rotation.is_clifford());
        assert_eq!(rotation.axis.iter().filter(|p| **p != Pauli::I).count(), 1);

        let qubit_index = rotation.axis.iter().position(|p| *p != Pauli::I).unwrap();
        match rotation.axis[qubit_index] {
            Pauli::I => unreachable!(),
            Pauli::X => Self {
                from_x: Pauli::X,
                from_y: Pauli::Z,
                from_z: Pauli::Y,
                qubit_index,
            },
            Pauli::Y => Self {
                from_x: Pauli::Z,
                from_y: Pauli::Y,
                from_z: Pauli::X,
                qubit_index,
            },
            Pauli::Z => Self {
                from_x: Pauli::Y,
                from_y: Pauli::X,
                from_z: Pauli::Z,
                qubit_index,
            },
        }
    }

    // Maps `rotation` to a new PauliRotation by Clifford conjugation.
    fn map(&self, op: &Operation) -> Operation {
        let mut axis: Axis = op.axis().clone();
        axis[self.qubit_index] = match axis[self.qubit_index] {
            Pauli::I => Pauli::I,
            Pauli::X => self.from_x,
            Pauli::Y => self.from_y,
            Pauli::Z => self.from_z,
        };
        match op {
            Operation::PauliRotation(r) => Operation::PauliRotation(PauliRotation {
                axis,
                angle: r.angle,
            }),
            Operation::Measurement(_) => Operation::Measurement(axis),
        }
    }
}

impl std::ops::Mul for SingleQubitCliffordConjugation {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        assert_eq!(self.qubit_index, rhs.qubit_index);

        let from_x = match self.from_x {
            Pauli::I => unreachable!(),
            Pauli::X => rhs.from_x,
            Pauli::Y => rhs.from_y,
            Pauli::Z => rhs.from_z,
        };
        let from_y = match self.from_y {
            Pauli::I => unreachable!(),
            Pauli::X => rhs.from_x,
            Pauli::Y => rhs.from_y,
            Pauli::Z => rhs.from_z,
        };
        let from_z = match self.from_z {
            Pauli::I => unreachable!(),
            Pauli::X => rhs.from_x,
            Pauli::Y => rhs.from_y,
            Pauli::Z => rhs.from_z,
        };
        Self {
            from_x,
            from_y,
            from_z,
            qubit_index: self.qubit_index,
        }
    }
}

pub fn lapbc_translation(ops: &Vec<Operation>) -> Vec<Operation> {
    let mut result = Vec::new();
    let mut clifford_rotations_for = Vec::<SingleQubitCliffordConjugation>::new();
    for i in 0..ops[0].axis().len() {
        clifford_rotations_for.push(SingleQubitCliffordConjugation::new_empty(i));
    }

    for op in ops {
        let mut target_qubit_indices =
            op.axis().iter().enumerate().filter_map(
                |(i, p)| {
                    if *p == Pauli::I {
                        None
                    } else {
                        Some(i)
                    }
                },
            );
        match op {
            Operation::PauliRotation(r) => {
                if op.is_single_qubit_clifford() {
                    let c = SingleQubitCliffordConjugation::new(r);
                    let target_qubit_index = target_qubit_indices.next().unwrap();
                    assert!(target_qubit_indices.next().is_none());

                    clifford_rotations_for[target_qubit_index] =
                        c * clifford_rotations_for[target_qubit_index].clone();
                } else {
                    let mut o = op.clone();
                    for target_qubit_index in target_qubit_indices {
                        o = clifford_rotations_for[target_qubit_index].map(&o);
                    }
                    result.push(o);
                }
            }
            Operation::Measurement(_) => {
                // We assume all measurements are single-qubit measurements.
                let target_qubit_index = target_qubit_indices.next().unwrap();
                assert!(target_qubit_indices.next().is_none());

                result.push(clifford_rotations_for[target_qubit_index].map(&op));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_axis(axis_string: &str) -> Axis {
        let v = axis_string
            .chars()
            .map(|c| match c {
                'I' => Pauli::I,
                'X' => Pauli::X,
                'Y' => Pauli::Y,
                'Z' => Pauli::Z,
                _ => unreachable!(),
            })
            .collect();
        Axis::new(v)
    }

    #[test]
    fn test_lapbc_translation_cx() {
        use Operation::Measurement as M;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new_clifford(new_axis("IZII"))),
            R(PauliRotation::new_clifford(new_axis("IIXI"))),
            R(PauliRotation::new_clifford(new_axis("IZXI"))),
            R(PauliRotation::new_pi_over_8(new_axis("ZIII"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIZI"))),
            M(new_axis("IIZI")),
        ];

        let result = lapbc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new_clifford(new_axis("IZXI"))),
                R(PauliRotation::new_pi_over_8(new_axis("ZIII"))),
                R(PauliRotation::new_pi_over_8(new_axis("IIYI"))),
                M(new_axis("IIYI"))
            ]
        );
    }

    #[test]
    fn test_lapbc_translation_tiny() {
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new_clifford(new_axis("IIIXI"))),
            R(PauliRotation::new_clifford(new_axis("IIIZI"))),
            R(PauliRotation::new_clifford(new_axis("IIZII"))),
            R(PauliRotation::new_clifford(new_axis("IIIXI"))),
            R(PauliRotation::new_clifford(new_axis("IIZXI"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIIZI"))),
        ];

        let result = lapbc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new_clifford(new_axis("IIZZI"))),
                R(PauliRotation::new_pi_over_8(new_axis("IIIXI"))),
            ]
        );
    }

    #[test]
    fn test_single_qubit_cliffor_conjugation() {
        use Pauli::*;
        let rotation = PauliRotation::new_clifford(new_axis("IXII"));
        let op = Operation::PauliRotation(rotation.clone());

        let c = SingleQubitCliffordConjugation::new(&rotation);
        assert_eq!(c.qubit_index, 1);
        assert_eq!(c.from_x, X);
        assert_eq!(c.from_y, Z);
        assert_eq!(c.from_z, Y);

        let c2 = c.clone() * SingleQubitCliffordConjugation::new(&rotation);
        assert_eq!(c2.qubit_index, 1);
        assert_eq!(c2.from_x, X);
        assert_eq!(c2.from_y, Y);
        assert_eq!(c2.from_z, Z);

        let mapped_op = c.map(&op);
        assert_eq!(mapped_op, op);

        let c3 =
            SingleQubitCliffordConjugation::new(&PauliRotation::new_clifford(new_axis("IYII")));
        assert_eq!(c3.map(&op), Operation::PauliRotation(PauliRotation::new_clifford(new_axis("IZII"))));

        let c4 = c * c3;
        assert_eq!(c4.qubit_index, 1);
        assert_eq!(c4.from_x, Z);
        assert_eq!(c4.from_y, X);
        assert_eq!(c4.from_z, Y);
    }
}
