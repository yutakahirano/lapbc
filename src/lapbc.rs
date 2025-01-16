use std::fmt;

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

#[derive(Clone, Debug, PartialEq)]
pub struct AxisPermutation {
    rotations: Vec<PauliRotation>,
}

impl AxisPermutation {
    #[cfg(test)]
    fn new(rotations: Vec<PauliRotation>) -> Self {
        assert!(rotations
            .iter()
            .all(|r| r.axis.iter().filter(|p| **p != Pauli::I).count() == 1));
        Self { rotations }
    }

    fn new_empty() -> Self {
        Self {
            rotations: Vec::new(),
        }
    }

    fn push(&mut self, rotation: PauliRotation) {
        assert!(rotation.axis.iter().filter(|p| **p != Pauli::I).count() == 1);
        self.rotations.push(rotation);
    }
}

impl fmt::Display for AxisPermutation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.rotations.is_empty() {
            return write!(f, "(empty)");
        }
        let n = self.rotations[0].axis.len();
        let mut axis = Axis::new(vec![Pauli::I; n]);
        for r in &self.rotations {
            for q in 0..n {
                if r.axis[q] != Pauli::I {
                    axis[q] = r.axis[q];
                }
            }
        }
        write!(f, "    ")?;
        axis.iter().try_for_each(|p| match p {
            Pauli::I => write!(f, "I"),
            Pauli::Y => write!(f, "H"),
            Pauli::Z => write!(f, "S"),
            _ => unreachable!(),
        })
    }
}

fn lapbc_compact_axis_permutation(commuting_ops: &[&Operation]) -> AxisPermutation {
    let mut perm = AxisPermutation::new_empty();

    if commuting_ops.is_empty() {
        return perm;
    }
    let n = commuting_ops[0].axis().len();
    assert!(n > 0);
    assert!(n < u32::MAX as usize);
    let mut axis = Axis::new(vec![Pauli::I; n]);
    for i in 0..commuting_ops.len() {
        let op = commuting_ops[i];

        // Skip operators that have multi-qubit support overlapping with preceding operators.
        if op.has_multi_qubit_support()
            && commuting_ops.iter().take(i).any(|other| {
                op.axis()
                    .has_overlapping_support_larger_than_one_qubit(other.axis())
            })
        {
            continue;
        }

        for q in 0..n {
            if op.axis()[q] != Pauli::I && op.axis()[q] != Pauli::Z {
                axis[q] = op.axis()[q];
            }
        }
    }

    assert!(n > 0);
    if n == 1 {
        use Pauli::*;
        match axis[0] {
            I | Z => {}
            X => {
                perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(0, n, Y)));
            }
            Y => {
                perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(0, n, Z)));
            }
        }
        return perm;
    }

    assert!(n >= 2);
    let mut x_count = 0_u32;
    let mut y_count = 0_u32;
    match axis[0] {
        Pauli::X => x_count += 1,
        Pauli::Y => y_count += 1,
        _ => {}
    }
    match axis[1] {
        Pauli::X => x_count += 1,
        Pauli::Y => y_count += 1,
        _ => {}
    }

    for i in 1..(n / 2) {
        match (axis[2 * i], axis[2 * i + 1]) {
            (Pauli::X, Pauli::Y) | (Pauli::Y, Pauli::X) => {
                x_count += 1;
                y_count += 1;
            }
            (Pauli::X, _) | (_, Pauli::X) => x_count += 1,
            (Pauli::Y, _) | (_, Pauli::Y) => y_count += 1,
            _ => {}
        }
    }
    if n % 2 == 1 {
        match axis[n - 1] {
            Pauli::X => x_count += 1,
            Pauli::Y => y_count += 1,
            _ => {}
        }
    }

    if y_count >= x_count {
        use Pauli::*;
        // Perform X <=> Y permutations.
        if axis[0] == Y {
            perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(0, n, Z)));
        }
        if axis[1] == Y {
            perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(1, n, Z)));
        }
        for i in 1..(n / 2) {
            if axis[2 * i] == Y {
                perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(2 * i, n, Z)));
            } else if axis[2 * i + 1] == Y {
                perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(2 * i + 1, n, Z)));
            }
        }
        if n % 2 == 1 && axis[n - 1] == Y {
            perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(n - 1, n, Z)));
        }
    } else {
        use Pauli::*;
        // Perform Z <=> X and X <=> Y permutations in this priority.
        if axis[0] == X {
            perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(0, n, Y)));
        } else if axis[0] == Y {
            perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(0, n, Z)));
        }
        if axis[1] == X {
            perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(1, n, Y)));
        } else if axis[1] == Y {
            perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(1, n, Z)));
        }
        for i in 1..(n / 2) {
            if axis[2 * i] == X {
                perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(2 * i, n, Y)));
            } else if axis[2 * i + 1] == X {
                perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(2 * i + 1, n, Y)));
            } else if axis[2 * i] == Y {
                perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(2 * i, n, Z)));
            } else if axis[2 * i + 1] == Y {
                perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(2 * i + 1, n, Z)));
            }
        }
        if n % 2 == 1 && axis[n - 1] == X {
            perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(n - 1, n, Y)));
        } else if n % 2 == 1 && axis[n - 1] == Y {
            perm.push(PauliRotation::new_clifford(Axis::new_with_pauli(n - 1, n, Z)));
        }
    }

    perm
}

#[derive(Clone, Debug, PartialEq)]
pub enum LapbcCompactOperation {
    Operation(Operation),
    AxisPermutation(AxisPermutation),
    #[allow(dead_code)]
    Noop,
}

impl std::fmt::Display for LapbcCompactOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use LapbcCompactOperation::*;
        match self {
            Operation(r) => write!(f, "{}", r),
            AxisPermutation(axis) => write!(f, "AxisPermutation({})", axis),
            Noop => write!(f, "Noop"),
        }
    }
}

impl LapbcCompactOperation {
    pub fn clocks(&self) -> u32 {
        match self {
            LapbcCompactOperation::Operation(_) => 1,
            LapbcCompactOperation::AxisPermutation(perm) => {
                if perm.rotations.iter().any(|r| r.axis.has_y()) {
                    3
                } else {
                    2
                }
            }
            LapbcCompactOperation::Noop => 1,
        }
    }
}

#[allow(dead_code)]
pub fn lapbc_compact_translation(ops: &Vec<Operation>) -> Vec<LapbcCompactOperation> {
    let mut ops = lapbc_translation(ops);
    if ops.is_empty() {
        return Vec::new();
    }

    let mut done = vec![false; ops.len()];
    let mut count = 0;
    let mut result = Vec::new();

    'outer: while count < ops.len() {
        let commuting_ops = ops
            .iter()
            .enumerate()
            .filter(|(i, _)| !done[*i])
            .filter(|(i, op)| {
                ops.iter()
                    .enumerate()
                    .take(*i)
                    .all(|(j, other)| done[j] || op.commutes_with(other))
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        assert!(!commuting_ops.is_empty());

        // If there is an operator with an axis consisting of only Z and I, perform it.
        let mut found = false;
        for i in &commuting_ops {
            let op = &ops[*i];
            if op.axis().has_only_z_and_i() {
                done[*i] = true;
                count += 1;
                result.push(LapbcCompactOperation::Operation(op.clone()));
                found = true;
            }
        }
        if found {
            continue 'outer;
        }

        // From now on, `commuting_ops` is a list of operators rather than indices.
        let commuting_ops = commuting_ops.iter().map(|i| &ops[*i]).collect::<Vec<_>>();
        let perm = lapbc_compact_axis_permutation(&commuting_ops);
        let clifford_rotations = &perm.rotations;
        for (i, op) in ops.iter_mut().enumerate() {
            if done[i] {
                continue;
            }
            match op {
                Operation::PauliRotation(r) => {
                    let mut rotation = r.clone();
                    for clifford_rotation in clifford_rotations.iter() {
                        rotation.axis.transform(&clifford_rotation.axis);
                    }
                    *op = Operation::PauliRotation(rotation);
                }
                Operation::Measurement(axis) => {
                    let mut a = axis.clone();
                    for clifford_rotation in clifford_rotations.iter() {
                        a.transform(&clifford_rotation.axis);
                    }
                    *op = Operation::Measurement(a);
                }
            }
        }
        result.push(LapbcCompactOperation::AxisPermutation(perm));
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
    fn test_lapbc_compact_axis_permutation_1() {
        use Operation::PauliRotation as R;
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("I")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(perm, AxisPermutation::new(vec![]));
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("X")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("Y")),])
            );
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("Y")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("Z")),])
            );
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("Z")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(perm, AxisPermutation::new(vec![]));
        }
    }

    #[test]
    fn test_lapbc_compact_axis_permutation_2() {
        use Operation::PauliRotation as R;
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("IZ")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(perm, AxisPermutation::new(vec![]));
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XZ")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("YI")),])
            );
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("ZI")),])
            );
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("YI")),
                    PauliRotation::new_clifford(new_axis("IY")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("ZI")),
                    PauliRotation::new_clifford(new_axis("IZ")),
                ])
            );
        }
    }

    #[test]
    fn test_lapbc_compact_axis_permutation_odd() {
        use Operation::PauliRotation as R;
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("IZZ")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(perm, AxisPermutation::new(vec![]));
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XZI")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("YII")),])
            );
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("ZZX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIY")),])
            );
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("ZZY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIZ")),])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XXX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("YII")),
                    PauliRotation::new_clifford(new_axis("IYI")),
                    PauliRotation::new_clifford(new_axis("IIY")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YXX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("ZII")),
                    PauliRotation::new_clifford(new_axis("IYI")),
                    PauliRotation::new_clifford(new_axis("IIY")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XYX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("YII")),
                    PauliRotation::new_clifford(new_axis("IZI")),
                    PauliRotation::new_clifford(new_axis("IIY")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XXY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("YII")),
                    PauliRotation::new_clifford(new_axis("IYI")),
                    PauliRotation::new_clifford(new_axis("IIZ")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YYY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("ZII")),
                    PauliRotation::new_clifford(new_axis("IZI")),
                    PauliRotation::new_clifford(new_axis("IIZ")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XYY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("IZI")),
                    PauliRotation::new_clifford(new_axis("IIZ")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YXY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("ZII")),
                    PauliRotation::new_clifford(new_axis("IIZ")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YYX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("ZII")),
                    PauliRotation::new_clifford(new_axis("IZI")),
                ])
            );
        }
    }

    #[test]
    fn test_lapbc_compact_axis_permutation_even() {
        use Operation::PauliRotation as R;
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("IZZI")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(perm, AxisPermutation::new(vec![]));
        }

        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XZIY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIIZ")),])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("ZZXX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIYI")),])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("ZZXY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIIZ")),])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("ZZYX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIZI")),])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("ZZYY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIZI")),])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XXYY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("YIII")),
                    PauliRotation::new_clifford(new_axis("IYII")),
                    PauliRotation::new_clifford(new_axis("IIZI")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YYXX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("ZIII")),
                    PauliRotation::new_clifford(new_axis("IZII")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XXXX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("YIII")),
                    PauliRotation::new_clifford(new_axis("IYII")),
                    PauliRotation::new_clifford(new_axis("IIYI")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XXYX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("YIII")),
                    PauliRotation::new_clifford(new_axis("IYII")),
                    PauliRotation::new_clifford(new_axis("IIIY")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("XXXY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("YIII")),
                    PauliRotation::new_clifford(new_axis("IYII")),
                    PauliRotation::new_clifford(new_axis("IIYI")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YYYY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("ZIII")),
                    PauliRotation::new_clifford(new_axis("IZII")),
                    PauliRotation::new_clifford(new_axis("IIZI")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YYXY")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("ZIII")),
                    PauliRotation::new_clifford(new_axis("IZII")),
                    PauliRotation::new_clifford(new_axis("IIIZ")),
                ])
            );
        }
        {
            let ops = [R(PauliRotation::new_clifford(new_axis("YYYX")))];
            let perm = lapbc_compact_axis_permutation(ops.iter().collect::<Vec<_>>().as_slice());
            assert_eq!(
                perm,
                AxisPermutation::new(vec![
                    PauliRotation::new_clifford(new_axis("ZIII")),
                    PauliRotation::new_clifford(new_axis("IZII")),
                    PauliRotation::new_clifford(new_axis("IIZI")),
                ])
            );
        }
    }

    #[test]
    fn test_lapbc_compact_translation_trivial() {
        use LapbcCompactOperation::Operation as O;
        use Operation::Measurement as M;
        use Operation::PauliRotation as R;

        let ops = vec![
            R(PauliRotation::new_clifford(new_axis("IZIZ"))),
            R(PauliRotation::new_pi_over_8(new_axis("IZII"))),
            R(PauliRotation::new_clifford(new_axis("IZZI"))),
            M(new_axis("IIIZ")),
        ];

        let result = lapbc_compact_translation(&ops);

        assert_eq!(
            result,
            vec![
                O(R(PauliRotation::new_clifford(new_axis("IZIZ")))),
                O(R(PauliRotation::new_pi_over_8(new_axis("IZII")))),
                O(R(PauliRotation::new_clifford(new_axis("IZZI")))),
                O(M(new_axis("IIIZ"))),
            ]
        );
    }

    #[test]
    fn test_lapbc_compact_translation_tiny() {
        use LapbcCompactOperation::AxisPermutation as A;
        use LapbcCompactOperation::Operation as O;
        use Operation::Measurement as M;
        use Operation::PauliRotation as R;

        let ops = vec![
            R(PauliRotation::new_pi_over_8(new_axis("XIII"))),
            R(PauliRotation::new_pi_over_8(new_axis("ZIII"))),
            R(PauliRotation::new_pi_over_8(new_axis("XIII"))),
            R(PauliRotation::new_pi_over_8(new_axis("IXII"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIXI"))),
            M(new_axis("IIIY")),
        ];

        let result = lapbc_compact_translation(&ops);
        let expected = vec![
            A(AxisPermutation::new(vec![
                PauliRotation::new_clifford(new_axis("YIII")),
                PauliRotation::new_clifford(new_axis("IYII")),
                PauliRotation::new_clifford(new_axis("IIYI")),
            ])),
            O(R(PauliRotation::new_pi_over_8(new_axis("ZIII")))),
            O(R(PauliRotation::new_pi_over_8(new_axis("IZII")))),
            O(R(PauliRotation::new_pi_over_8(new_axis("IIZI")))),
            A(AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIIZ"))])),
            A(AxisPermutation::new(vec![
                PauliRotation::new_clifford(new_axis("YIII")),
                PauliRotation::new_clifford(new_axis("IIIY")),
            ])),
            O(R(PauliRotation::new_pi_over_8(new_axis("ZIII")))),
            O(M(new_axis("IIIZ"))),
            A(AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("YIII"))])),
            O(R(PauliRotation::new_pi_over_8(new_axis("ZIII")))),
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_lapbc_compact_translation_small() {
        use LapbcCompactOperation::AxisPermutation as A;
        use LapbcCompactOperation::Operation as O;
        use Operation::Measurement as M;
        use Operation::PauliRotation as R;

        let ops = vec![
            R(PauliRotation::new_pi_over_8(new_axis("XIIIII"))),
            R(PauliRotation::new_pi_over_8(new_axis("IXIIII"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIXIII"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIIXII"))),
            R(PauliRotation::new_clifford(new_axis("XIIIIZ"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIIIZI"))),
            R(PauliRotation::new_pi_over_8(new_axis("IIYIII"))),
            R(PauliRotation::new_pi_over_8(new_axis("ZIIIIX"))),
            M(new_axis("IIIIIX")),
        ];

        let result = lapbc_compact_translation(&ops);
        let expected = vec![
            O(R(PauliRotation::new_pi_over_8(new_axis("IIIIZI")))),
            A(AxisPermutation::new(vec![
                PauliRotation::new_clifford(new_axis("YIIIII")),
                PauliRotation::new_clifford(new_axis("IYIIII")),
                PauliRotation::new_clifford(new_axis("IIYIII")),
            ])),
            O(R(PauliRotation::new_pi_over_8(new_axis("ZIIIII")))),
            O(R(PauliRotation::new_pi_over_8(new_axis("IZIIII")))),
            O(R(PauliRotation::new_pi_over_8(new_axis("IIZIII")))),
            O(R(PauliRotation::new_clifford(new_axis("ZIIIIZ")))),
            A(AxisPermutation::new(vec![
                PauliRotation::new_clifford(new_axis("YIIIII")),
                PauliRotation::new_clifford(new_axis("IIIYII")),
                PauliRotation::new_clifford(new_axis("IIIIIY")),
            ])),
            O(R(PauliRotation::new_pi_over_8(new_axis("IIIZII")))),
            O(R(PauliRotation::new_pi_over_8(new_axis("ZIIIIZ")))),
            O(M(new_axis("IIIIIZ"))),
            A(AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIZIII"))])),
            A(AxisPermutation::new(vec![PauliRotation::new_clifford(new_axis("IIYIII"))])),
            O(R(PauliRotation::new_pi_over_8(new_axis("IIZIII")))),
        ];
        assert_eq!(result[6], expected[6]);
        assert_eq!(result[7], expected[7]);
        assert_eq!(result[8], expected[8]);
        assert_eq!(result[9], expected[9]);
        assert_eq!(result[10], expected[10]);
        assert_eq!(result, expected);
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
