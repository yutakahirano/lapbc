use super::pbc::Angle;
use super::pbc::Axis;
use super::pbc::Mod8;
use super::pbc::Operation;
use super::pbc::Pauli;
use super::pbc::PauliRotation;
use super::pbc::Sign;

// Representing a single-qubit Clifford that maps Pauli to Pauli through conjugation.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SingleQubitCliffordConjugation {
    from_x: (Pauli, Sign),
    from_y: (Pauli, Sign),
    from_z: (Pauli, Sign),
    qubit_index: usize,
}

impl SingleQubitCliffordConjugation {
    fn new_empty(qubit_index: usize) -> Self {
        Self {
            from_x: (Pauli::X, Sign::Plus),
            from_y: (Pauli::Y, Sign::Plus),
            from_z: (Pauli::Z, Sign::Plus),
            qubit_index,
        }
    }
    fn new_pauli(qubit_index: usize, pauli: Pauli) -> Self {
        match pauli {
            Pauli::I => unreachable!(),
            Pauli::X => Self {
                from_x: (Pauli::X, Sign::Plus),
                from_y: (Pauli::Y, Sign::Minus),
                from_z: (Pauli::Z, Sign::Minus),
                qubit_index,
            },
            Pauli::Y => Self {
                from_x: (Pauli::X, Sign::Minus),
                from_y: (Pauli::Y, Sign::Plus),
                from_z: (Pauli::Z, Sign::Minus),
                qubit_index,
            },
            Pauli::Z => Self {
                from_x: (Pauli::X, Sign::Minus),
                from_y: (Pauli::Y, Sign::Minus),
                from_z: (Pauli::Z, Sign::Plus),
                qubit_index,
            },
        }
    }

    fn new(rotation: &PauliRotation) -> Self {
        assert!(rotation.is_clifford());
        assert_eq!(rotation.axis.iter().filter(|p| **p != Pauli::I).count(), 1);

        let qubit_index = rotation.axis.iter().position(|p| *p != Pauli::I).unwrap();

        let sign = match rotation.angle {
            Angle::PiOver8(Mod8::Zero) => {
                return Self::new_empty(qubit_index);
            }
            Angle::PiOver8(Mod8::Four) => {
                return Self::new_pauli(qubit_index, rotation.axis[qubit_index]);
            }
            Angle::PiOver8(Mod8::Two) => Sign::Plus,
            Angle::PiOver8(Mod8::Six) => Sign::Minus,
            _ => unreachable!(),
        };

        match rotation.axis[qubit_index] {
            Pauli::I => unreachable!(),
            Pauli::X => Self {
                from_x: (Pauli::X, Sign::Plus),
                from_y: (Pauli::Z, Sign::Minus * sign),
                from_z: (Pauli::Y, Sign::Plus * sign),
                qubit_index,
            },
            Pauli::Y => Self {
                from_x: (Pauli::Z, Sign::Plus * sign),
                from_y: (Pauli::Y, Sign::Plus),
                from_z: (Pauli::X, Sign::Minus * sign),
                qubit_index,
            },
            Pauli::Z => Self {
                from_x: (Pauli::Y, Sign::Minus * sign),
                from_y: (Pauli::X, Sign::Plus * sign),
                from_z: (Pauli::Z, Sign::Plus),
                qubit_index,
            },
        }
    }

    // Maps `rotation` to a new PauliRotation by Clifford conjugation.
    fn map(&self, op: &Operation) -> Operation {
        let mut axis: Axis = op.axis().clone();
        let (pauli, sign) = match axis[self.qubit_index] {
            Pauli::I => (Pauli::I, Sign::Plus),
            Pauli::X => self.from_x,
            Pauli::Y => self.from_y,
            Pauli::Z => self.from_z,
        };

        axis[self.qubit_index] = pauli;
        match op {
            Operation::PauliRotation(r) => Operation::PauliRotation(PauliRotation {
                axis,
                angle: r.angle * sign,
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
            (Pauli::I, _) => unreachable!(),
            (Pauli::X, sign) => (rhs.from_x.0, rhs.from_x.1 * sign),
            (Pauli::Y, sign) => (rhs.from_y.0, rhs.from_y.1 * sign),
            (Pauli::Z, sign) => (rhs.from_z.0, rhs.from_z.1 * sign),
        };
        let from_y = match self.from_y {
            (Pauli::I, _) => unreachable!(),
            (Pauli::X, sign) => (rhs.from_x.0, rhs.from_x.1 * sign),
            (Pauli::Y, sign) => (rhs.from_y.0, rhs.from_y.1 * sign),
            (Pauli::Z, sign) => (rhs.from_z.0, rhs.from_z.1 * sign),
        };
        let from_z = match self.from_z {
            (Pauli::I, _) => unreachable!(),
            (Pauli::X, sign) => (rhs.from_x.0, rhs.from_x.1 * sign),
            (Pauli::Y, sign) => (rhs.from_y.0, rhs.from_y.1 * sign),
            (Pauli::Z, sign) => (rhs.from_z.0, rhs.from_z.1 * sign),
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
    fn test_lapbc_translation_one_qubit_x() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("XIIII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("YIIII"), -PiOver8(One))),
            R(PauliRotation::new(new_axis("XIIII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IXIII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IZIII"), -PiOver8(One))),
            R(PauliRotation::new(new_axis("IXIII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIXII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIYII"), -PiOver8(One))),
            R(PauliRotation::new(new_axis("IIXII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIXI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIZI"), -PiOver8(One))),
            R(PauliRotation::new(new_axis("IIIXI"), -PiOver8(Two))),
        ];

        let result = lapbc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new(new_axis("ZIIII"), -PiOver8(One))),
                R(PauliRotation::new(new_axis("IYIII"), PiOver8(One))),
                R(PauliRotation::new(new_axis("IIZII"), PiOver8(One))),
                R(PauliRotation::new(new_axis("IIIYI"), -PiOver8(One))),
            ]
        );
    }

    #[test]
    fn test_lapbc_translation_one_qubit_y() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("YIIII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("ZIIII"), -PiOver8(One))),
            R(PauliRotation::new(new_axis("YIIII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IYIII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IXIII"), -PiOver8(One))),
            R(PauliRotation::new(new_axis("IYIII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIYII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIZII"), -PiOver8(One))),
            R(PauliRotation::new(new_axis("IIYII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIYI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIXI"), -PiOver8(One))),
            R(PauliRotation::new(new_axis("IIIYI"), -PiOver8(Two))),
        ];

        let result = lapbc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new(new_axis("XIIII"), -PiOver8(One))),
                R(PauliRotation::new(new_axis("IZIII"), PiOver8(One))),
                R(PauliRotation::new(new_axis("IIXII"), PiOver8(One))),
                R(PauliRotation::new(new_axis("IIIZI"), -PiOver8(One))),
            ]
        );
    }

    #[test]
    fn test_lapbc_translation_one_qubit_z() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("ZIIII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("XIIII"), PiOver8(One))),
            R(PauliRotation::new(new_axis("ZIIII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IZIII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IYIII"), PiOver8(One))),
            R(PauliRotation::new(new_axis("IZIII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIZII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIXII"), PiOver8(One))),
            R(PauliRotation::new(new_axis("IIZII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIZI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIYI"), PiOver8(One))),
            R(PauliRotation::new(new_axis("IIIZI"), -PiOver8(Two))),
        ];

        let result = lapbc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new(new_axis("YIIII"), PiOver8(One))),
                R(PauliRotation::new(new_axis("IXIII"), -PiOver8(One))),
                R(PauliRotation::new(new_axis("IIYII"), -PiOver8(One))),
                R(PauliRotation::new(new_axis("IIIXI"), PiOver8(One))),
            ]
        );
    }

    #[test]
    fn test_lapbc_translation_cx() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::Measurement as M;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("IZII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIXI"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IZXI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("ZIII"), -PiOver8(One))),
            R(PauliRotation::new(new_axis("IIZI"), PiOver8(One))),
            M(new_axis("IIZI")),
        ];

        let result = lapbc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new(new_axis("IZXI"), PiOver8(Two))),
                R(PauliRotation::new(new_axis("ZIII"), PiOver8(Seven))),
                R(PauliRotation::new(new_axis("IIYI"), PiOver8(Seven))),
                M(new_axis("IIYI"))
            ]
        );
    }

    #[test]
    fn test_lapbc_translation_tiny() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("IIIXI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIZI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIZII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIXI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIZXI"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIZI"), -PiOver8(One))),
        ];

        let result = lapbc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new(new_axis("IIZZI"), PiOver8(Six))),
                R(PauliRotation::new(new_axis("IIIXI"), PiOver8(Seven))),
            ]
        );
    }

    #[test]
    fn test_single_qubit_clifford_conjugation_identity() {
        use Pauli::*;
        use Sign::*;

        let id_1 = SingleQubitCliffordConjugation::new_empty(1);
        assert_eq!(id_1.qubit_index, 1);
        assert_eq!(id_1.from_x, (X, Plus));
        assert_eq!(id_1.from_y, (Y, Plus));
        assert_eq!(id_1.from_z, (Z, Plus));

        let id_2 = SingleQubitCliffordConjugation::new_empty(2);
        assert_eq!(id_2.qubit_index, 2);
        assert_eq!(id_2.from_x, (X, Plus));
        assert_eq!(id_2.from_y, (Y, Plus));
        assert_eq!(id_2.from_z, (Z, Plus));

        let id_3 = SingleQubitCliffordConjugation::new(&PauliRotation::new(
            new_axis("IIIZ"),
            Angle::PiOver8(Mod8::Zero),
        ));
        assert_eq!(id_3.qubit_index, 3);
        assert_eq!(id_3.from_x, (X, Plus));
        assert_eq!(id_3.from_y, (Y, Plus));
        assert_eq!(id_3.from_z, (Z, Plus));
    }

    #[test]
    fn test_single_qubit_clifford_conjugation_pauli() {
        use Pauli::*;
        use Sign::*;

        let angle = Angle::PiOver8(Mod8::Four);
        let x1 = SingleQubitCliffordConjugation::new(&PauliRotation::new(new_axis("IXII"), angle));
        let y2 = SingleQubitCliffordConjugation::new(&PauliRotation::new(new_axis("IIYI"), angle));
        let z3 = SingleQubitCliffordConjugation::new(&PauliRotation::new(new_axis("IIIZ"), angle));

        assert_eq!(x1.qubit_index, 1);
        assert_eq!(x1.from_x, (X, Plus));
        assert_eq!(x1.from_y, (Y, Minus));
        assert_eq!(x1.from_z, (Z, Minus));

        assert_eq!(y2.qubit_index, 2);
        assert_eq!(y2.from_x, (X, Minus));
        assert_eq!(y2.from_y, (Y, Plus));
        assert_eq!(y2.from_z, (Z, Minus));

        assert_eq!(z3.qubit_index, 3);
        assert_eq!(z3.from_x, (X, Minus));
        assert_eq!(z3.from_y, (Y, Minus));
        assert_eq!(z3.from_z, (Z, Plus));

        assert_eq!(x1.clone() * x1, SingleQubitCliffordConjugation::new_empty(1));
        assert_eq!(y2.clone() * y2, SingleQubitCliffordConjugation::new_empty(2));
        assert_eq!(z3.clone() * z3, SingleQubitCliffordConjugation::new_empty(3));
    }

    #[test]
    fn test_single_qubit_clifford_conjugation() {
        use Angle::PiOver8;
        use Mod8::*;
        use Pauli::*;
        use Sign::*;

        let rotation = PauliRotation::new(new_axis("IXII"), -PiOver8(Two));
        let op = Operation::PauliRotation(rotation.clone());

        let c = SingleQubitCliffordConjugation::new(&rotation);
        assert_eq!(c.qubit_index, 1);
        assert_eq!(c.from_x, (X, Plus));
        assert_eq!(c.from_y, (Z, Plus));
        assert_eq!(c.from_z, (Y, Minus));

        assert_eq!(c.clone() * c.clone(), SingleQubitCliffordConjugation::new(
            &PauliRotation::new(new_axis("IXII"), PiOver8(Four))));

        let mapped_op = c.map(&op);
        assert_eq!(mapped_op, op);

        let c3 = SingleQubitCliffordConjugation::new(&PauliRotation::new(
            new_axis("IYII"),
            PiOver8(Two),
        ));
        assert_eq!(
            c3.map(&op),
            Operation::PauliRotation(PauliRotation::new(new_axis("IZII"), PiOver8(Six)))
        );

        let c4 = c * c3;
        assert_eq!(c4.qubit_index, 1);
        assert_eq!(c4.from_x, (Z, Plus));
        assert_eq!(c4.from_y, (X, Minus));
        assert_eq!(c4.from_z, (Y, Minus));
    }
}
