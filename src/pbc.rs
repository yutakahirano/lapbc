use serde::Deserialize;
use serde::Serialize;

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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Sign {
    Plus,
    Minus,
}

impl std::ops::Mul<Sign> for Sign {
    type Output = Self;

    fn mul(self, sign: Sign) -> Self {
        if self == sign {
            Sign::Plus
        } else {
            Sign::Minus
        }
    }
}

impl std::ops::MulAssign<Sign> for Sign {
    fn mul_assign(&mut self, sign: Sign) {
        *self = *self * sign;
    }
}

impl std::ops::Neg for Sign {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Sign::Plus => Sign::Minus,
            Sign::Minus => Sign::Plus,
        }
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

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Serialize)]
pub enum Mod8 {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
}

impl Mod8 {
    pub fn from(n: u32) -> Self {
        match n % 8 {
            0 => Mod8::Zero,
            1 => Mod8::One,
            2 => Mod8::Two,
            3 => Mod8::Three,
            4 => Mod8::Four,
            5 => Mod8::Five,
            6 => Mod8::Six,
            7 => Mod8::Seven,
            _ => unreachable!(),
        }
    }

    pub fn to_u32(&self) -> u32 {
        match self {
            Mod8::Zero => 0,
            Mod8::One => 1,
            Mod8::Two => 2,
            Mod8::Three => 3,
            Mod8::Four => 4,
            Mod8::Five => 5,
            Mod8::Six => 6,
            Mod8::Seven => 7,
        }
    }
}

impl std::ops::Neg for Mod8 {
    type Output = Self;
    fn neg(self) -> Self {
        match self {
            Mod8::Zero => Mod8::Zero,
            Mod8::One => Mod8::Seven,
            Mod8::Two => Mod8::Six,
            Mod8::Three => Mod8::Five,
            Mod8::Four => Mod8::Four,
            Mod8::Five => Mod8::Three,
            Mod8::Six => Mod8::Two,
            Mod8::Seven => Mod8::One,
        }
    }
}

// Angle represents the rotation angle of a Pauli rotation.
// Since exp(-i * pi * P) = I for every Pauli operator P, we only need to consider angle modulo pi.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub enum Angle {
    // PiOver8(n) represents n * pi / 8.
    PiOver8(Mod8),
    Arbitrary(f64),
}

impl std::ops::Neg for Angle {
    type Output = Self;
    fn neg(self) -> Self {
        match self {
            Angle::PiOver8(n) => Angle::PiOver8(-n),
            Angle::Arbitrary(angle) => Angle::Arbitrary(-angle),
        }
    }
}

impl std::ops::Mul<Sign> for Angle {
    type Output = Self;

    fn mul(self, sign: Sign) -> Self {
        match sign {
            Sign::Plus => self,
            Sign::Minus => -self,
        }
    }
}

// PauliRotation represents a Pauli rotation consisting of a rotation axis and an angle.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PauliRotation {
    pub axis: Axis,
    pub angle: Angle,
}

impl PauliRotation {
    pub fn is_clifford(&self) -> bool {
        use Angle::*;
        use Mod8::*;
        match self.angle {
            PiOver8(Zero) | PiOver8(Two) | PiOver8(Four) | PiOver8(Six) => true,
            PiOver8(One) | PiOver8(Three) | PiOver8(Five) | PiOver8(Seven) | Arbitrary(_) => false,
        }
    }

    pub fn transform(&mut self, clifford_rotation: &PauliRotation) {
        use Mod8::*;
        use Pauli::*;
        assert!(clifford_rotation.is_clifford());
        if self.axis.commutes_with(&clifford_rotation.axis) {
            return;
        }

        let mut sign = match clifford_rotation.angle {
            Angle::PiOver8(Zero) => {
                return;
            }
            Angle::PiOver8(Four) => {
                self.angle = -self.angle;
                return;
            }
            Angle::PiOver8(Two) => Sign::Plus,
            Angle::PiOver8(Six) => Sign::Minus,
            _ => unreachable!(),
        };
        for (a, b) in self.axis.axis.iter_mut().zip(clifford_rotation.axis.iter()) {
            match (*a, *b) {
                (I, I) | (X, X) | (Y, Y) | (Z, Z) | (I, _) | (_, I) => {}
                (X, Y) | (Y, Z) | (Z, X) => {}
                (Y, X) | (Z, Y) | (X, Z) => {
                    sign = -sign;
                }
            }
            *a = *a * *b;
        }
        if sign == Sign::Minus {
            self.angle = -self.angle;
        }
    }

    pub fn new(axis: Axis, angle: Angle) -> Self {
        PauliRotation { axis, angle }
    }
}

impl std::fmt::Display for PauliRotation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use Angle::*;
        match self.angle {
            PiOver8(n) => {
                write!(f, "axis: {}, angle: PiOver8({})", self.axis, n.to_u32())
            }
            Arbitrary(angle) => {
                write!(f, "axis: {}, angle: {}", self.axis, angle)
            }
        }
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
pub fn spc_translation(ops: &[Operation]) -> Vec<Operation> {
    use Angle::*;
    use Mod8::*;
    let mut result = Vec::new();
    let mut clifford_rotations = Vec::new();
    for op in ops {
        match op {
            Operation::PauliRotation(r) => {
                let angle = match r.angle {
                    PiOver8(Zero) => {
                        continue;
                    }
                    PiOver8(Two) | PiOver8(Four) | PiOver8(Six) => {
                        clifford_rotations.push(r.clone());
                        continue;
                    }
                    PiOver8(Three) => {
                        clifford_rotations.push(PauliRotation::new(r.axis.clone(), PiOver8(Four)));
                        -PiOver8(One)
                    }
                    PiOver8(Five) => {
                        clifford_rotations.push(PauliRotation::new(r.axis.clone(), PiOver8(Four)));
                        PiOver8(One)
                    }
                    PiOver8(One) | PiOver8(Seven) | Arbitrary(_) => r.angle,
                };
                assert!(matches!(
                    angle,
                    PiOver8(One) | PiOver8(Seven) | Arbitrary(_)
                ));
                let mut rotation = PauliRotation::new(r.axis.clone(), angle);
                for clifford_rotation in clifford_rotations.iter().rev() {
                    rotation.transform(clifford_rotation);
                }
                result.push(Operation::PauliRotation(rotation));
            }
            Operation::Measurement(axis) => {
                let mut r = PauliRotation::new(axis.clone(), Angle::PiOver8(Mod8::Four));
                for clifford_rotation in clifford_rotations.iter().rev() {
                    r.transform(clifford_rotation);
                }
                result.push(Operation::Measurement(r.axis.clone()));
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
    fn test_tranform_rotation() {
        use Angle::PiOver8;
        use Mod8::*;
        {
            let mut r = PauliRotation::new(new_axis("XXYZ"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("IIII"), PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("XXYZ"), PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("XXYZ"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("IIII"), -PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("XXYZ"), PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("X"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Z"), PiOver8(Four)));

            assert_eq!(r, PauliRotation::new(new_axis("X"), -PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("X"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Y"), PiOver8(Four)));

            assert_eq!(r, PauliRotation::new(new_axis("X"), -PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("X"), -PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Z"), PiOver8(Four)));

            assert_eq!(r, PauliRotation::new(new_axis("X"), PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("X"), -PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Y"), PiOver8(Four)));

            assert_eq!(r, PauliRotation::new(new_axis("X"), PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("X"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Z"), PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("Y"), -PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("Y"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Z"), PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("X"), PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("X"), -PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Z"), PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("Y"), PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("Y"), -PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Z"), PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("X"), -PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("X"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Z"), -PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("Y"), PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("Y"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("Z"), -PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("X"), -PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("XXYZ"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("IIZI"), PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("XXXZ"), PiOver8(One)));
        }

        {
            let mut r = PauliRotation::new(new_axis("IZZI"), PiOver8(One));
            r.transform(&PauliRotation::new(new_axis("IIXI"), PiOver8(Two)));

            assert_eq!(r, PauliRotation::new(new_axis("IZYI"), PiOver8(One)));
        }
    }

    #[test]
    fn test_spc_translation_trivial() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("IIIXI"), PiOver8(One))),
            R(PauliRotation::new(new_axis("IIIYI"), PiOver8(Two))),
        ];

        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![R(PauliRotation::new(new_axis("IIIXI"), PiOver8(One)))],
        );
    }

    #[test]
    fn test_spc_translation_commuting() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("IIZXI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("ZIIII"), PiOver8(Six))),
            R(PauliRotation::new(new_axis("IIXYI"), PiOver8(One))),
        ];

        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![R(PauliRotation::new(new_axis("IIXYI"), PiOver8(One)))],
        );
    }

    #[test]
    fn test_spc_translation_tiny_1() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("Z"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("Y"), PiOver8(One))),
        ];

        // Given S * X * Sdg = Y, S * exp(i * theta * X) * Sdg = exp(i * theta * Y). Therefore,
        //    exp(i * Y * pi / 8) * S
        //  = S * exp(i * X * pi / 8) * Sdg * S
        //  = S * exp(i * X * pi / 8).
        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![R(PauliRotation::new(new_axis("X"), PiOver8(One)))],
        );
    }

    #[test]
    fn test_spc_translation_tiny_2() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("Z"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("X"), PiOver8(One))),
        ];

        // Given S * Y * Sdg = -X, S * exp(i * theta * Y) * Sdg = exp(-i * theta * X). Therefore,
        //    exp(i * X * pi / 8) * S
        //  = S * exp(-i * Y * pi / 8) * Sdg * S
        //  = S * exp(-i * Y * pi / 8).
        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![R(PauliRotation::new(new_axis("Y"), -PiOver8(One)))],
        );
    }

    #[test]
    fn test_spc_translation_tiny_3() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("Z"), PiOver8(Six))),
            R(PauliRotation::new(new_axis("Y"), PiOver8(One))),
        ];

        // Given S * X * Sdg = Y, S * exp(i * theta * X) * Sdg = exp(i * theta * Y). Therefore,
        //    exp(i * Y * pi / 8) * Sdg
        //  = S * exp(i * X * pi / 8) * Sdg * Sdg
        //  = S * exp(i * X * pi / 8) * Z
        //  = S * Z * exp(-i * X * pi / 8)
        //  = Sdg * exp(-i * X * pi / 8)
        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![R(PauliRotation::new(new_axis("X"), -PiOver8(One)))],
        );
    }

    #[test]
    fn test_spc_translation_cx() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::Measurement as M;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("IZII"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIXI"), -PiOver8(Two))),
            R(PauliRotation::new(new_axis("IZXI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("ZIII"), PiOver8(One))),
            R(PauliRotation::new(new_axis("IIZI"), PiOver8(One))),
            M(new_axis("IIZI")),
        ];

        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new(new_axis("ZIII"), PiOver8(One))),
                R(PauliRotation::new(new_axis("IZZI"), PiOver8(One))),
                M(new_axis("IZZI"))
            ]
        );
    }

    #[test]
    fn test_spc_translation_small() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("IIIXI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIZI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIZII"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIXI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIZXI"), PiOver8(Two))),
            R(PauliRotation::new(new_axis("IIIZI"), PiOver8(One))),
        ];

        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![R(PauliRotation::new(new_axis("IIZYI"), -PiOver8(One)))]
        );
    }

    #[test]
    fn test_spc_translation_with_various_angles() {
        use Angle::PiOver8;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let ops = vec![
            R(PauliRotation::new(new_axis("IIIZI"), PiOver8(Six))),
            R(PauliRotation::new(new_axis("IZIXI"), PiOver8(Five))),
            R(PauliRotation::new(new_axis("IXZII"), PiOver8(Three))),
            R(PauliRotation::new(new_axis("IXIII"), PiOver8(Seven))),
            R(PauliRotation::new(new_axis("IIIYI"), PiOver8(Three))),
        ];

        let result = spc_translation(&ops);
        assert_eq!(
            result,
            vec![
                R(PauliRotation::new(new_axis("IZIYI"), PiOver8(One))),
                R(PauliRotation::new(new_axis("IXZII"), PiOver8(One))),
                R(PauliRotation::new(new_axis("IXIII"), PiOver8(One))),
                R(PauliRotation::new(new_axis("IIIXI"), PiOver8(Seven))),
            ]
        );
    }
}
