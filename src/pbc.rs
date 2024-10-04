// One-qubit Pauli operator.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
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

// Axis is a multi-qubit Pauli operator and it represents the rotation axis of a Pauli rotation.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    pub fn has_y(&self) -> bool {
        self.axis.iter().any(|p| *p == Pauli::Y)
    }
    pub fn has_only_z_and_i(&self) -> bool {
        self.axis.iter().all(|p| *p == Pauli::Z || *p == Pauli::I)
    }

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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Angle {
    Zero,
    PiOver2,
    PiOver4,
    PiOver8,
    Arbitrary(f64),
}

// PauliRotation represents a Pauli rotation consisting of a rotation axis and an angle.
#[derive(Debug, Clone, PartialEq)]
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

// Operator represents an operator in the Pauli-based Computation.
#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    PauliRotation(PauliRotation),
    Measurement(Axis),
}

impl Operator {
    pub fn is_non_clifford_rotation_or_measurement(&self) -> bool {
        match self {
            Operator::PauliRotation(r) => !r.is_clifford(),
            Operator::Measurement(..) => true,
        }
    }

    pub fn is_single_qubit_clifford(&self) -> bool {
        match self {
            Operator::PauliRotation(r) => r.is_clifford() && self.has_single_qubit_support(),
            Operator::Measurement(..) => false,
        }
    }

    pub fn is_multi_qubit_clifford(&self) -> bool {
        match self {
            Operator::PauliRotation(r) => r.is_clifford() && self.has_multi_qubit_support(),
            Operator::Measurement(..) => false,
        }
    }

    pub fn axis(&self) -> &Axis {
        match self {
            Operator::PauliRotation(r) => &r.axis,
            Operator::Measurement(a) => a,
        }
    }

    pub fn has_single_qubit_support(&self) -> bool {
        self.axis().iter().filter(|p| **p != Pauli::I).count() == 1
    }
    pub fn has_multi_qubit_support(&self) -> bool {
        self.axis().iter().filter(|p| **p != Pauli::I).count() > 1
    }

    pub fn commutes_with(&self, other: &Operator) -> bool {
        self.axis().commutes_with(other.axis())
    }
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Operator::PauliRotation(r) => write!(f, "PauliRotation({})", r),
            Operator::Measurement(axis) => write!(f, "Measurement({})", axis),
        }
    }
}

// Performs the SPC translation.
pub fn spc_translation(ops: &Vec<Operator>) -> Vec<Operator> {
    let mut result = Vec::new();
    let mut clifford_rotations = Vec::new();
    for op in ops {
        match op {
            Operator::PauliRotation(r) => {
                if r.is_clifford() {
                    clifford_rotations.push(r.clone());
                } else {
                    let mut rotation = r.clone();
                    for clifford_rotation in clifford_rotations.iter().rev() {
                        rotation.axis.transform(&clifford_rotation.axis);
                    }
                    result.push(Operator::PauliRotation(rotation));
                }
            }
            Operator::Measurement(axis) => {
                let mut a = axis.clone();
                for clifford_rotation in clifford_rotations.iter().rev() {
                    a.transform(&clifford_rotation.axis);
                }
                result.push(Operator::Measurement(a));
            }
        }
    }

    result
}

// Given an Operator `op`, returns a pair of a list of clifford operators and the number of
// additional clocks.
// The cliford operators are needed to translate the axis of `op` so that it consists of only
// Z and I Pauli operators. The order of the list matters. For example, if `op`'s axis is
// `Y`, then the clifford operators will be [Z, Y] where Z maps Y to X and Y maps X to Z.
fn spc_compact_translation_one(op: &Operator) -> (Vec<PauliRotation>, u32) {
    let mut clifford_operations: Vec<PauliRotation> = Vec::new();
    let axis = op.axis();
    let y_count = axis.iter().filter(|p| **p == Pauli::Y).count();
    let mut additional_clocks = 0;
    if y_count == 0 {
        // Do nothing.
    } else if y_count % 2 == 1 {
        // Apply a XY permutation.
        let rotation_axis = Axis::new(
            axis.iter()
                .map(|p| match p {
                    Pauli::I => Pauli::I,
                    Pauli::X => Pauli::I,
                    Pauli::Y => Pauli::Z,
                    Pauli::Z => Pauli::I,
                })
                .collect(),
        );
        clifford_operations.push(PauliRotation::new_clifford(rotation_axis));
        additional_clocks += 1;
    } else {
        // Apply two XY permutations.
        let mut first = true;
        let rotation_axis = Axis::new(
            axis.iter()
                .map(|p| match p {
                    Pauli::I => Pauli::I,
                    Pauli::X => Pauli::I,
                    Pauli::Y => {
                        if first {
                            first = false;
                            Pauli::Z
                        } else {
                            Pauli::I
                        }
                    }
                    Pauli::Z => Pauli::I,
                })
                .collect(),
        );
        clifford_operations.push(PauliRotation::new_clifford(rotation_axis));
        let mut first = true;
        let rotation_axis = Axis::new(
            axis.iter()
                .map(|p| match p {
                    Pauli::I => Pauli::I,
                    Pauli::X => Pauli::I,
                    Pauli::Y => {
                        if first {
                            first = false;
                            Pauli::I
                        } else {
                            Pauli::Z
                        }
                    }
                    Pauli::Z => Pauli::I,
                })
                .collect(),
        );
        clifford_operations.push(PauliRotation::new_clifford(rotation_axis));
        additional_clocks += 2;
    }
    let mut has_xy = false;
    let mut needs_two_rounds = false;

    for i in 0..axis.len() {
        if i % 2 == 1 {
            continue;
        }

        let a = axis[i];
        let a_xy = a == Pauli::X || a == Pauli::Y;
        if a_xy {
            clifford_operations.push(PauliRotation::new_clifford(Axis::new_with_pauli(
                i,
                axis.len(),
                Pauli::Y,
            )));
            has_xy = true;
        }

        if i == axis.len() - 1 {
            break;
        }

        let b = axis[i + 1];
        let b_xy = b == Pauli::X || b == Pauli::Y;
        if b_xy {
            has_xy = true;
            clifford_operations.push(PauliRotation::new_clifford(Axis::new_with_pauli(
                i + 1,
                axis.len(),
                Pauli::Y,
            )));
        }

        if i == 0 {
            continue;
        }

        if a_xy && b_xy {
            needs_two_rounds = true;
        }
    }

    if needs_two_rounds {
        additional_clocks += 6;
    } else if has_xy {
        additional_clocks += 3;
    }

    (clifford_operations, additional_clocks)
}

// Performs the compact SPC translation.
pub fn spc_compact_translation(ops: &Vec<Operator>) -> Vec<(Operator, u32)> {
    let ops = spc_translation(ops);
    let mut result = Vec::new();
    let mut clifford_rotations: Vec<PauliRotation> = Vec::new();

    for op in &ops {
        let mut op = op.clone();
        // Apply existing clifford ops.
        match op {
            Operator::PauliRotation(r) => {
                assert!(!r.is_clifford());
                let mut rotation = r.clone();
                for clifford_rotation in clifford_rotations.iter() {
                    rotation.axis.transform(&clifford_rotation.axis);
                }
                op = Operator::PauliRotation(rotation);
            }
            Operator::Measurement(axis) => {
                let mut a = axis.clone();
                for clifford_rotation in clifford_rotations.iter() {
                    a.transform(&clifford_rotation.axis);
                }
                op = Operator::Measurement(a);
            }
        }

        let (additional_cliffords, additional_clocks) = spc_compact_translation_one(&op);
        clifford_rotations.extend(additional_cliffords);
        result.push((op, additional_clocks));
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
        use Operator::Measurement as M;
        use Operator::PauliRotation as R;
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
        use Operator::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new_clifford(new_axis("IIIXI"))),
            R(PauliRotation::new_clifford(new_axis("IIIZI"))),
            R(PauliRotation::new_clifford(new_axis("IIZII"))),
            R(PauliRotation::new_clifford(new_axis("IIIXI"))),
            R(PauliRotation::new_clifford(new_axis("IIZXI"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIIZI"))),
        ];

        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![R(PauliRotation::new_pi_over_8(new_axis("IIZYI")))]
        );
    }
    #[test]
    fn test_spc_compact_translation_one_trivial() {
        use Operator::PauliRotation as R;

        let (clifford_ops, additional_clocks) =
            spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("IIIIII"))));
        assert_eq!(clifford_ops.len(), 0);
        assert_eq!(additional_clocks, 0);
    }

    #[test]
    fn test_spc_compact_translation_one_with_no_additional_clocks() {
        use Operator::PauliRotation as R;
        let (clifford_ops, additional_clocks) =
            spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("ZZIZZZ"))));
        assert_eq!(clifford_ops.len(), 0);
        assert_eq!(additional_clocks, 0);
    }

    #[test]
    fn test_spc_compact_translation_one_with_xz_permutation() {
        use Operator::PauliRotation as R;
        {
            let (clifford_ops, additional_clocks) =
                spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("IXIIII"))));
            assert_eq!(
                clifford_ops,
                vec![PauliRotation::new_clifford(new_axis("IYIIII"))]
            );
            assert_eq!(additional_clocks, 3);
        }

        {
            let (clifford_ops, additional_clocks) =
                spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("XXIIII"))));
            assert_eq!(
                clifford_ops,
                vec![
                    PauliRotation::new_clifford(new_axis("YIIIII")),
                    PauliRotation::new_clifford(new_axis("IYIIII"))
                ]
            );
            assert_eq!(additional_clocks, 3);
        }

        {
            let (clifford_ops, additional_clocks) =
                spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("XXXIII"))));
            assert_eq!(
                clifford_ops,
                vec![
                    PauliRotation::new_clifford(new_axis("YIIIII")),
                    PauliRotation::new_clifford(new_axis("IYIIII")),
                    PauliRotation::new_clifford(new_axis("IIYIII")),
                ]
            );
            assert_eq!(additional_clocks, 3);
        }

        {
            let (clifford_ops, additional_clocks) =
                spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("XXXXII"))));
            assert_eq!(
                clifford_ops,
                vec![
                    PauliRotation::new_clifford(new_axis("YIIIII")),
                    PauliRotation::new_clifford(new_axis("IYIIII")),
                    PauliRotation::new_clifford(new_axis("IIYIII")),
                    PauliRotation::new_clifford(new_axis("IIIYII")),
                ]
            );
            assert_eq!(additional_clocks, 6);
        }
    }

    #[test]
    fn test_spc_compact_translation_one_with_xy_permutation() {
        use Operator::PauliRotation as R;
        {
            let (clifford_ops, additional_clocks) =
                spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("IYIIII"))));
            assert_eq!(
                clifford_ops,
                vec![
                    PauliRotation::new_clifford(new_axis("IZIIII")),
                    PauliRotation::new_clifford(new_axis("IYIIII"))
                ]
            );
            assert_eq!(additional_clocks, 4);
        }

        {
            let (clifford_ops, additional_clocks) =
                spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("YYIIII"))));
            assert_eq!(
                clifford_ops,
                vec![
                    PauliRotation::new_clifford(new_axis("ZIIIII")),
                    PauliRotation::new_clifford(new_axis("IZIIII")),
                    PauliRotation::new_clifford(new_axis("YIIIII")),
                    PauliRotation::new_clifford(new_axis("IYIIII"))
                ]
            );
            assert_eq!(additional_clocks, 5);
        }

        {
            let (clifford_ops, additional_clocks) =
                spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("YYYIII"))));
            assert_eq!(
                clifford_ops,
                vec![
                    PauliRotation::new_clifford(new_axis("ZZZIII")),
                    PauliRotation::new_clifford(new_axis("YIIIII")),
                    PauliRotation::new_clifford(new_axis("IYIIII")),
                    PauliRotation::new_clifford(new_axis("IIYIII")),
                ]
            );
            assert_eq!(additional_clocks, 4);
        }

        {
            let (clifford_ops, additional_clocks) =
                spc_compact_translation_one(&R(PauliRotation::new_pi_over_8(new_axis("YYYYII"))));
            assert_eq!(
                clifford_ops,
                vec![
                    PauliRotation::new_clifford(new_axis("ZIIIII")),
                    PauliRotation::new_clifford(new_axis("IZZZII")),
                    PauliRotation::new_clifford(new_axis("YIIIII")),
                    PauliRotation::new_clifford(new_axis("IYIIII")),
                    PauliRotation::new_clifford(new_axis("IIYIII")),
                    PauliRotation::new_clifford(new_axis("IIIYII")),
                ]
            );
            assert_eq!(additional_clocks, 8);
        }
    }

    #[test]
    fn test_spc_compact_translation_one_qubit() {
        use Operator::PauliRotation as R;

        let ops = vec![
            R(PauliRotation::new_pi_over_8(new_axis("X"))),
            R(PauliRotation::new_pi_over_8(new_axis("Y"))),
            R(PauliRotation::new_pi_over_8(new_axis("X"))),
            R(PauliRotation::new_pi_over_8(new_axis("X"))),
        ];

        let result = spc_compact_translation(&ops);
        assert_eq!(
            result,
            vec![
                (R(PauliRotation::new_pi_over_8(new_axis("X"))), 3),
                (R(PauliRotation::new_pi_over_8(new_axis("Y"))), 4),
                (R(PauliRotation::new_pi_over_8(new_axis("X"))), 3),
                (R(PauliRotation::new_pi_over_8(new_axis("Z"))), 0),
            ]
        );
    }

    #[test]
    fn test_spc_compact_translation() {
        use Operator::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new_pi_over_8(new_axis("IIIXII"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIIXIZ"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIYYII"))),
            R(PauliRotation::new_pi_over_8(new_axis("IXIIYX"))),
            R(PauliRotation::new_pi_over_8(new_axis("XXXXXX"))),
            R(PauliRotation::new_pi_over_8(new_axis("ZIIIIZ"))),
        ];

        let result = spc_compact_translation(&ops);
        assert_eq!(
            result,
            vec![
                (R(PauliRotation::new_pi_over_8(new_axis("IIIXII"))), 3),
                (R(PauliRotation::new_pi_over_8(new_axis("IIIZIZ"))), 0),
                (R(PauliRotation::new_pi_over_8(new_axis("IIYYII"))), 8),
                (R(PauliRotation::new_pi_over_8(new_axis("IXIIYX"))), 7),
                (R(PauliRotation::new_pi_over_8(new_axis("XZYXYZ"))), 8),
                (R(PauliRotation::new_pi_over_8(new_axis("XIIIIX"))), 3),
            ]
        );
    }
}
