use serde::Serialize;
use serde::Deserialize;

// One-qubit Pauli operation.
#[derive(Debug, Clone, Copy, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl std::ops::Mul for Pauli {
    type Output = Pauli;

    fn mul(self, other: Self) -> Self {
        match &self {
            Pauli::I => other,
            Pauli::X => match other {
                Pauli::I => Pauli::X,
                Pauli::X => Pauli::I,
                Pauli::Y => Pauli::Z,
                Pauli::Z => Pauli::Y,
            },
            Pauli::Y => match other {
                Pauli::I => Pauli::Y,
                Pauli::X => Pauli::Z,
                Pauli::Y => Pauli::I,
                Pauli::Z => Pauli::X,
            },
            Pauli::Z => match other {
                Pauli::I => Pauli::Z,
                Pauli::X => Pauli::Y,
                Pauli::Y => Pauli::X,
                Pauli::Z => Pauli::I,
            },
        }
    }
}

impl Pauli {
    pub fn commutes_with(&self, other: &Pauli) -> bool {
        *self == Pauli::I || *other == Pauli::I || self == other
    }
}

impl std::fmt::Display for Pauli {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Pauli::I => 'I',
                Pauli::X => 'X',
                Pauli::Y => 'Y',
                Pauli::Z => 'Z',
            }
        )
    }
}

// Axis is a multi-qubit Pauli operation and it represents the rotation axis of a Pauli rotation.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct Axis {
    axis: Vec<Pauli>,
}

impl Axis {
    pub fn new(axis: Vec<Pauli>) -> Self {
        Axis { axis }
    }

    pub fn new_with_pauli(index: usize, size: usize, pauli: Pauli) -> Self {
        assert!(index < size);
        let mut axis = vec![Pauli::I; size];
        axis[index] = pauli;
        Axis::new(axis)
    }

    pub fn commutes_with(&self, other: &Axis) -> bool {
        assert_eq!(self.len(), other.len());
        let count = self
            .axis
            .iter()
            .zip(other.axis.iter())
            .filter(|(a, b)| !a.commutes_with(b))
            .count();
        count % 2 == 0
    }

    pub fn transform(&mut self, other: &Axis) {
        assert_eq!(self.len(), other.len());
        if self.commutes_with(other) {
            return;
        }
        for (a, b) in self.axis.iter_mut().zip(other.axis.iter()) {
            *a = *a * *b;
        }
    }

    pub fn iter(&self) -> std::slice::Iter<Pauli> {
        self.axis.iter()
    }

    pub fn len(&self) -> usize {
        self.axis.len()
    }

    #[allow(dead_code)]
    pub fn has_x(&self) -> bool {
        self.axis.iter().any(|p| *p == Pauli::X)
    }
    #[allow(dead_code)]
    pub fn has_y(&self) -> bool {
        self.axis.iter().any(|p| *p == Pauli::Y)
    }
    #[allow(dead_code)]
    pub fn has_only_z_and_i(&self) -> bool {
        self.axis.iter().all(|p| *p == Pauli::Z || *p == Pauli::I)
    }

    #[allow(dead_code)]
    pub fn has_overlapping_support_larger_than_one_qubit(&self, other: &Axis) -> bool {
        assert_eq!(self.axis.len(), other.axis.len());
        use Pauli::I;
        self.axis
            .iter()
            .zip(other.axis.iter())
            .filter(|(a, b)| a != &&I && b != &&I)
            .count()
            > 1
    }
}

impl std::ops::Index<usize> for Axis {
    type Output = Pauli;

    fn index(&self, index: usize) -> &Self::Output {
        &self.axis[index]
    }
}

impl std::ops::IndexMut<usize> for Axis {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.axis[index]
    }
}

impl std::fmt::Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.axis
                .iter()
                .map(|p| match p {
                    Pauli::I => 'I',
                    Pauli::X => 'X',
                    Pauli::Y => 'Y',
                    Pauli::Z => 'Z',
                })
                .collect::<String>()
        )
    }
}

// Angle represents the rotation angle of a Pauli rotation.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub enum Angle {
    Zero,
    PiOver2,
    PiOver4,
    PiOver8,
    Arbitrary(f64),
}

// PauliRotation represents a Pauli rotation consisting of a rotation axis and an angle.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PauliRotation {
    pub axis: Axis,
    pub angle: Angle,
}

impl PauliRotation {
    pub fn is_clifford(&self) -> bool {
        self.angle == Angle::PiOver4
    }

    pub fn new(axis: Axis, angle: Angle) -> Self {
        PauliRotation { axis, angle }
    }

    pub fn new_clifford(axis: Axis) -> Self {
        PauliRotation::new(axis, Angle::PiOver4)
    }

    #[cfg(test)]
    pub fn new_pi_over_8(axis: Axis) -> Self {
        PauliRotation::new(axis, Angle::PiOver8)
    }
}

impl std::fmt::Display for PauliRotation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "axis: {}, angle: {:?}", self.axis, self.angle)
    }
}

// Operation represents an operation in the Pauli-based Computation.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Operation {
    PauliRotation(PauliRotation),
    Measurement(Axis),
}

impl Operation {
    #[allow(dead_code)]
    pub fn is_non_clifford_rotation_or_measurement(&self) -> bool {
        match self {
            Operation::PauliRotation(r) => !r.is_clifford(),
            Operation::Measurement(..) => true,
        }
    }

    #[allow(dead_code)]
    pub fn is_single_qubit_clifford(&self) -> bool {
        match self {
            Operation::PauliRotation(r) => r.is_clifford() && self.has_single_qubit_support(),
            Operation::Measurement(..) => false,
        }
    }

    #[allow(dead_code)]
    pub fn is_multi_qubit_clifford(&self) -> bool {
        match self {
            Operation::PauliRotation(r) => r.is_clifford() && self.has_multi_qubit_support(),
            Operation::Measurement(..) => false,
        }
    }

    #[allow(dead_code)]
    pub fn axis(&self) -> &Axis {
        match self {
            Operation::PauliRotation(r) => &r.axis,
            Operation::Measurement(a) => a,
        }
    }

    #[allow(dead_code)]
    pub fn has_single_qubit_support(&self) -> bool {
        self.axis().iter().filter(|p| **p != Pauli::I).count() == 1
    }
    #[allow(dead_code)]
    pub fn has_multi_qubit_support(&self) -> bool {
        self.axis().iter().filter(|p| **p != Pauli::I).count() > 1
    }

    #[allow(dead_code)]
    pub fn commutes_with(&self, other: &Operation) -> bool {
        self.axis().commutes_with(other.axis())
    }
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Operation::PauliRotation(r) => write!(f, "PauliRotation({})", r),
            Operation::Measurement(axis) => write!(f, "Measurement({})", axis),
        }
    }
}

// Performs the SPC translation.
pub fn spc_translation(ops: &Vec<Operation>) -> Vec<Operation> {
    let mut result = Vec::new();
    let mut clifford_rotations = Vec::new();
    for op in ops {
        match op {
            Operation::PauliRotation(r) => {
                if r.is_clifford() {
                    clifford_rotations.push(r.clone());
                } else {
                    let mut rotation = r.clone();
                    for clifford_rotation in clifford_rotations.iter().rev() {
                        rotation.axis.transform(&clifford_rotation.axis);
                    }
                    result.push(Operation::PauliRotation(rotation));
                }
            }
            Operation::Measurement(axis) => {
                let mut a = axis.clone();
                for clifford_rotation in clifford_rotations.iter().rev() {
                    a.transform(&clifford_rotation.axis);
                }
                result.push(Operation::Measurement(a));
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
    fn test_pauli_product() {
        use Pauli::*;
        assert_eq!(I * I, I);
        assert_eq!(I * X, X);
        assert_eq!(I * Y, Y);
        assert_eq!(I * Z, Z);

        assert_eq!(X * I, X);
        assert_eq!(X * X, I);
        assert_eq!(X * Y, Z);
        assert_eq!(X * Z, Y);

        assert_eq!(Y * I, Y);
        assert_eq!(Y * X, Z);
        assert_eq!(Y * Y, I);
        assert_eq!(Y * Z, X);

        assert_eq!(Z * I, Z);
        assert_eq!(Z * X, Y);
        assert_eq!(Z * Y, X);
        assert_eq!(Z * Z, I);
    }
    #[test]
    fn test_commutes_with() {
        use Pauli::*;
        assert!(I.commutes_with(&I));
        assert!(I.commutes_with(&X));
        assert!(I.commutes_with(&Y));
        assert!(I.commutes_with(&Z));

        assert!(X.commutes_with(&I));
        assert!(X.commutes_with(&X));
        assert!(!X.commutes_with(&Y));
        assert!(!X.commutes_with(&Z));

        assert!(Y.commutes_with(&I));
        assert!(!Y.commutes_with(&X));
        assert!(Y.commutes_with(&Y));
        assert!(!Y.commutes_with(&Z));

        assert!(Z.commutes_with(&I));
        assert!(!Z.commutes_with(&X));
        assert!(!Z.commutes_with(&Y));
        assert!(Z.commutes_with(&Z));
    }

    #[test]
    fn test_commutes_with_axis() {
        assert!(new_axis("IIII").commutes_with(&new_axis("XYZI")));
        assert!(new_axis("XYXY").commutes_with(&new_axis("YZYX")));
        assert!(new_axis("XYZ").commutes_with(&new_axis("YYY")));
        assert!(!new_axis("XYZ").commutes_with(&new_axis("YYZ")));
        assert!(!new_axis("IXYZ").commutes_with(&new_axis("IYYZ")));
    }

    #[test]
    fn test_axis_has_x() {
        assert!(!new_axis("").has_x());

        assert!(!new_axis("I").has_x());
        assert!(new_axis("X").has_x());
        assert!(!new_axis("Y").has_x());
        assert!(!new_axis("Z").has_x());

        assert!(!new_axis("II").has_x());
        assert!(new_axis("XI").has_x());
        assert!(!new_axis("YI").has_x());
        assert!(!new_axis("ZI").has_x());
        assert!(new_axis("IX").has_x());
        assert!(new_axis("XX").has_x());
        assert!(new_axis("YX").has_x());
        assert!(new_axis("ZX").has_x());
        assert!(!new_axis("IY").has_x());
        assert!(new_axis("XY").has_x());
        assert!(!new_axis("YY").has_x());
        assert!(!new_axis("ZY").has_x());
        assert!(!new_axis("IZ").has_x());
        assert!(new_axis("XZ").has_x());
        assert!(!new_axis("YZ").has_x());
        assert!(!new_axis("ZZ").has_x());
    }

    #[test]
    fn test_axis_has_y() {
        assert!(!new_axis("").has_y());

        assert!(!new_axis("I").has_y());
        assert!(!new_axis("X").has_y());
        assert!(new_axis("Y").has_y());
        assert!(!new_axis("Z").has_y());

        assert!(!new_axis("II").has_y());
        assert!(!new_axis("XI").has_y());
        assert!(new_axis("YI").has_y());
        assert!(!new_axis("ZI").has_y());
        assert!(!new_axis("IX").has_y());
        assert!(!new_axis("XX").has_y());
        assert!(new_axis("YX").has_y());
        assert!(!new_axis("ZX").has_y());
        assert!(new_axis("IY").has_y());
        assert!(new_axis("XY").has_y());
        assert!(new_axis("YY").has_y());
        assert!(new_axis("ZY").has_y());
        assert!(!new_axis("IZ").has_y());
        assert!(!new_axis("XZ").has_y());
        assert!(new_axis("YZ").has_y());
        assert!(!new_axis("ZZ").has_y());
    }

    #[test]
    fn test_axis_has_only_z_and_i() {
        assert!(new_axis("").has_only_z_and_i());

        assert!(new_axis("I").has_only_z_and_i());
        assert!(!new_axis("X").has_only_z_and_i());
        assert!(!new_axis("Y").has_only_z_and_i());
        assert!(new_axis("Z").has_only_z_and_i());

        assert!(new_axis("II").has_only_z_and_i());
        assert!(!new_axis("XI").has_only_z_and_i());
        assert!(!new_axis("YI").has_only_z_and_i());
        assert!(new_axis("ZI").has_only_z_and_i());
        assert!(!new_axis("IX").has_only_z_and_i());
        assert!(!new_axis("XX").has_only_z_and_i());
        assert!(!new_axis("YX").has_only_z_and_i());
        assert!(!new_axis("ZX").has_only_z_and_i());
        assert!(!new_axis("IY").has_only_z_and_i());
        assert!(!new_axis("XY").has_only_z_and_i());
        assert!(!new_axis("YY").has_only_z_and_i());
        assert!(!new_axis("ZY").has_only_z_and_i());
        assert!(new_axis("IZ").has_only_z_and_i());
        assert!(!new_axis("XZ").has_only_z_and_i());
        assert!(!new_axis("YZ").has_only_z_and_i());
        assert!(new_axis("ZZ").has_only_z_and_i());
    }

    #[test]
    fn test_has_overlapping_support_larger_than_one_qubit() {
        assert!(
            !new_axis("IIIII").has_overlapping_support_larger_than_one_qubit(&new_axis("IIIII"))
        );
        assert!(
            !new_axis("IIIII").has_overlapping_support_larger_than_one_qubit(&new_axis("XXXXX"))
        );
        assert!(
            !new_axis("IXIXX").has_overlapping_support_larger_than_one_qubit(&new_axis("YIYII"))
        );
        assert!(
            !new_axis("IXIXX").has_overlapping_support_larger_than_one_qubit(&new_axis("YYYII"))
        );

        assert!(new_axis("IXIXX").has_overlapping_support_larger_than_one_qubit(&new_axis("YYYIX")));
        assert!(new_axis("IXZII").has_overlapping_support_larger_than_one_qubit(&new_axis("IXZII")));
        assert!(new_axis("XXXXX").has_overlapping_support_larger_than_one_qubit(&new_axis("YYYYY")));
    }

    #[test]
    fn test_tranform_axis() {
        {
            let mut axis = new_axis("XXYZ");
            axis.transform(&new_axis("IIII"));

            assert_eq!(axis, new_axis("XXYZ"));
        }

        {
            let mut axis = new_axis("XXYZ");
            axis.transform(&new_axis("YYYY"));

            assert_eq!(axis, new_axis("ZZIX"));
        }

        {
            let mut axis = new_axis("XXYZ");
            axis.transform(&new_axis("IIZI"));

            assert_eq!(axis, new_axis("XXXZ"));
        }

        {
            let mut axis = new_axis("IZZI");
            axis.transform(&new_axis("IIXI"));

            assert_eq!(axis, new_axis("IZYI"));
        }
    }

    #[test]
    fn test_spc_translation_cx() {
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

        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new_pi_over_8(new_axis("ZIII"))),
                R(PauliRotation::new_pi_over_8(new_axis("IZZI"))),
                M(new_axis("IZZI"))
            ]
        );
    }

    #[test]
    fn test_spc_translation_tiny() {
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new_clifford(new_axis("IIIXI"))),
            R(PauliRotation::new_clifford(new_axis("IIIZI"))),
            R(PauliRotation::new_clifford(new_axis("IIZII"))),
            R(PauliRotation::new_clifford(new_axis("IIIXI"))),
            R(PauliRotation::new_clifford(new_axis("IIZXI"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIIZI"))),
        ];

        let result = spc_translation(&ops);
        assert_eq!(result, vec![R(PauliRotation::new_pi_over_8(new_axis("IIZYI")))]);
    }
}
