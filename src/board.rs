use std::collections::VecDeque;
use std::ops::{Index, IndexMut};
use std::{collections::HashMap, ops::Range};

use crate::mapping::{DataQubitMapping, Qubit};
use crate::pbc::{Angle, Operator, Pauli, PauliRotation};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct OperationId {
    id: u32,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum BoardOccupancy {
    Vacant,
    LatticeSurgery(OperationId),
    IdleDataQubit,
    DataQubitInOperation(OperationId),
    YInitialization,
    YMeasurement,
    MagicStateDistillation(OperationId),
}

pub struct Board {
    width: u32,
    height: u32,
    code_distance: u32,

    data_qubit_mapping: HashMap<Qubit, (u32, u32)>,
    occupancy: Vec<BoardOccupancy>,
    cycle_after_last_operation_at: Vec<u32>,
    cycle: u32,
    current_operation_id: OperationId,
}

struct Map2D<T: Clone + Default> {
    width: u32,
    height: u32,
    map: Vec<T>,
}

struct AncillaAvailability {
    map: Map2D<bool>,
}

impl OperationId {
    fn increment(&mut self) {
        self.id += 1;
    }
}

fn y_initialization_cost(distance: u32) -> u32 {
    distance + y_measurement_cost(distance)
}

fn y_measurement_cost(distance: u32) -> u32 {
    (distance / 2) + 2
}

impl Board {
    pub fn new(mapping: DataQubitMapping, code_distance: u32) -> Self {
        let occupancy = vec![];
        let last_operation_cycle = vec![];
        let current_operation_id = OperationId { id: 0 };

        Self {
            width: mapping.width,
            height: mapping.height,
            code_distance,
            data_qubit_mapping: mapping.iter().map(|(x, y, q)| (*q, (*x, *y))).collect(),
            occupancy,
            cycle_after_last_operation_at: last_operation_cycle,
            cycle: 0,
            current_operation_id,
        }
    }
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    fn issue_operation_id(&mut self) -> OperationId {
        let id = self.current_operation_id;
        self.current_operation_id.increment();
        id
    }

    pub fn schedule(&mut self, op: &Operator) -> bool {
        match op {
            Operator::Measurement(axis) => {
                let mut target: Option<Qubit> = None;
                let mut pauli_axis: Option<Pauli> = None;
                for (i, a) in axis.iter().enumerate() {
                    if *a == Pauli::I {
                        continue;
                    }
                    assert!(target.is_none());
                    assert!(pauli_axis.is_none());
                    target = Some(Qubit::new(i));
                    pauli_axis = Some(*a);
                }

                if target.is_none() {
                    // There is nothing to do.
                    return true;
                }

                self.schedule_mesuarement(target.unwrap(), pauli_axis.unwrap())
            }
            Operator::PauliRotation(rotation) => self.schedule_rotation(rotation),
        }
    }

    fn schedule_mesuarement(&mut self, qubit: Qubit, axis: Pauli) -> bool {
        let (x, y) = self.data_qubit_mapping[&qubit];

        let duration = match axis {
            Pauli::X => 1,
            Pauli::Y => (self.code_distance + 1) / 2,
            Pauli::Z => 1,
            Pauli::I => return true,
        };

        self.ensure_board_occupancy(self.cycle + duration);
        if self.has_schedule_at_or_after(qubit, self.cycle) {
            return false;
        }

        let operation_id = self.issue_operation_id();
        for c in self.cycle..self.cycle + duration {
            let o = BoardOccupancy::DataQubitInOperation(operation_id);
            self.set_occupancy(x, y, c, o);
        }
        self.set_cycle_after_last_operation_at(qubit, self.cycle + self.cycle + duration);

        true
    }

    fn schedule_rotation(&mut self, rotation: &PauliRotation) -> bool {
        match rotation.angle {
            Angle::Zero => true,
            Angle::PiOver2 => true,
            Angle::PiOver4 => self.schedule_pi_over_4_rotation(rotation),
            Angle::PiOver8 => self.schedule_pi_over_8_rotation(rotation),
            Angle::Arbitrary(..) => panic!("Not implemented yet"),
        }
    }

    fn schedule_pi_over_4_rotation(&mut self, rotation: &PauliRotation) -> bool {
        let support_size = rotation.axis.iter().filter(|a| **a != Pauli::I).count();
        if support_size == 1 {
            let target_position = rotation.axis.iter().position(|a| *a != Pauli::I).unwrap();
            let target = Qubit::new(target_position);
            self.schedule_single_qubit_pi_over_4_rotation(target, rotation.axis[target_position])
        } else if support_size == 2 {
            let mut it = rotation
                .axis
                .iter()
                .enumerate()
                .filter(|(_, a)| **a != Pauli::I);
            let position1 = it.next().unwrap().0;
            let position2 = it.next().unwrap().0;
            assert!(it.next().is_none());

            self.schedule_two_qubit_pi_over_4_rotation(
                Qubit::new(position1),
                rotation.axis[position1],
                Qubit::new(position2),
                rotation.axis[position2],
            )
        } else {
            unimplemented!("schedule_pi_over_4_rotation: support size > 2");
        }
    }

    fn schedule_single_qubit_pi_over_4_rotation(&mut self, qubit: Qubit, axis: Pauli) -> bool {
        if self.has_schedule_at_or_after(qubit, self.cycle) {
            return false;
        }

        let (x, y) = self.data_qubit_mapping[&qubit];
        let cycle = self.cycle;
        let distance = self.code_distance;

        self.ensure_board_occupancy(cycle + distance + y_initialization_cost(distance));
        match axis {
            Pauli::I => return true,
            Pauli::X => {
                let mut candidates = vec![];
                if x > 0 {
                    candidates.push((x - 1, y));
                }
                if x < self.width - 1 {
                    candidates.push((x + 1, y));
                }
                self.schedule_pi_over_4_rotation_internal(&[(qubit, x, y)], &[(x, y)], &candidates)
            }
            Pauli::Z => {
                let mut candidates = vec![];
                if y > 0 {
                    candidates.push((x, y - 1));
                }
                if y < self.height - 1 {
                    candidates.push((x, y + 1));
                }
                self.schedule_pi_over_4_rotation_internal(&[(qubit, x, y)], &[(x, y)], &candidates)
            }
            Pauli::Y => panic!("Not implemented"),
        }
    }

    fn schedule_two_qubit_pi_over_4_rotation(
        &mut self,
        q1: Qubit,
        axis1: Pauli,
        q2: Qubit,
        axis2: Pauli,
    ) -> bool {
        let cycle = self.cycle;
        let distance = self.code_distance;
        let y_initialization_cost = y_initialization_cost(distance);
        if self.has_schedule_at_or_after(q1, cycle) || self.has_schedule_at_or_after(q2, cycle) {
            return false;
        }

        let (x1, y1) = self.data_qubit_mapping[&q1];
        let (x2, y2) = self.data_qubit_mapping[&q2];

        self.ensure_board_occupancy(cycle + distance + y_initialization_cost);
        let map = AncillaAvailability::new(self, cycle..cycle + distance);
        let path = Self::path_between((x1, y1, axis1), (x2, y2, axis2), &map);
        let path = match path {
            Some(p) => p,
            None => return false,
        };

        let mut adjacent_ancilla_candidates: Vec<(u32, u32)> = vec![];
        for &(x, y) in &path {
            if (x, y) == (x1, y1) || (x, y) == (x2, y2) {
                continue;
            }
            let mut candidates = vec![];
            if x > 0 {
                candidates.push((x - 1, y));
            }
            if x + 1 < self.width {
                candidates.push((x + 1, y));
            }
            if y > 0 {
                candidates.push((x, y - 1));
            }
            if y + 1 < self.height {
                candidates.push((x, y + 1));
            }
            adjacent_ancilla_candidates.extend(
                candidates
                    .iter()
                    .filter(|&(cx, cy)| !path.contains(&(*cx, *cy))),
            );
        }

        self.schedule_pi_over_4_rotation_internal(
            &[(q1, x1, y1), (q2, x2, y2)],
            &path,
            &adjacent_ancilla_candidates,
        )
    }

    fn schedule_pi_over_4_rotation_internal(
        &mut self,
        targets: &[(Qubit, u32, u32)],
        path: &[(u32, u32)],
        ancilla_candidates: &[(u32, u32)],
    ) -> bool {
        let cycle = self.cycle;
        let distance = self.code_distance;
        let y_initialization_cost = y_initialization_cost(distance);
        let id = self.issue_operation_id();

        let mut found = false;

        if cycle >= y_initialization_cost {
            for &(x, y) in ancilla_candidates {
                if self.is_vacant(x, y, cycle - y_initialization_cost..cycle + distance) {
                    for c in cycle - y_initialization_cost..cycle {
                        self.set_occupancy(x, y, c, BoardOccupancy::YInitialization);
                    }
                    for c in cycle..cycle + distance {
                        self.set_occupancy(x, y, c, BoardOccupancy::LatticeSurgery(id));
                    }
                    for (q, _, _) in targets {
                        self.set_cycle_after_last_operation_at(*q, cycle + distance);
                    }
                    found = true;
                    break;
                }
            }
        }

        if !found {
            for &(x, y) in ancilla_candidates {
                if self.is_vacant(x, y, cycle..cycle + distance + y_initialization_cost) {
                    for c in cycle..cycle + distance {
                        self.set_occupancy(x, y, c, BoardOccupancy::LatticeSurgery(id));
                    }
                    for c in cycle + distance..cycle + distance + y_measurement_cost(distance) {
                        self.set_occupancy(x, y, c, BoardOccupancy::YMeasurement);
                    }
                    for (q, _, _) in targets {
                        self.set_cycle_after_last_operation_at(*q, cycle + distance);
                    }
                    found = true;
                    break;
                }
            }
        }

        if !found {
            return false;
        }

        for &(x, y) in path {
            let in_targets = targets.iter().any(|&(_, xt, yt)| xt == x && yt == y);
            let occupancy = if in_targets {
                BoardOccupancy::DataQubitInOperation(id)
            } else {
                BoardOccupancy::LatticeSurgery(id)
            };
            for c in cycle..cycle + distance {
                self.set_occupancy(x, y, c, occupancy);
            }
        }

        true
    }

    // Returns a path between `(x1, y1)` and `(x2, y2)`.
    // The two endpoints require specific connectivity constraints specified by `p1` and `p2`.
    // `p1` and `p2` are either Pauli::X, Pauli::Y, or Pauli::Z.
    // Note that when `p1` is Pauli::Y, then the result is not a strict path because it requires
    // adjacent three ancilla qubits. The same applies to `p2`.
    fn path_between(
        (x1, y1, p1): (u32, u32, Pauli),
        (x2, y2, p2): (u32, u32, Pauli),
        available: &AncillaAvailability,
    ) -> Option<Vec<(u32, u32)>> {
        let mut q = VecDeque::new();
        let width = available.width();
        let height = available.height();

        let mut history: Map2D<Option<(u32, u32)>> = Map2D::new_with_default(width, height);
        let mut src_y_history: Vec<[(u32, u32); 3]> = vec![];
        let mut dest_y_history: Vec<[(u32, u32); 3]> = vec![];

        match p1 {
            Pauli::I => {
                unreachable!("path_between: Pauli::I");
            }
            Pauli::X => {
                if x1 > 0 && available[(x1 - 1, y1)] {
                    q.push_back(((x1 - 1, y1), (x1, y1)));
                }
                if x1 + 1 < width && available[(x1 + 1, y1)] {
                    q.push_back(((x1 + 1, y1), (x1, y1)));
                }
            }
            Pauli::Z => {
                if y1 > 0 && available[(x1, y1 - 1)] {
                    q.push_back(((x1, y1 - 1), (x1, y1)));
                }
                if y1 + 1 < height && available[(x1, y1 + 1)] {
                    q.push_back(((x1, y1 + 1), (x1, y1)));
                }
            }
            Pauli::Y => {
                // We use this to avoid integer underflow. In any case, a value with underflow will
                // not be used because it will be filtered out by `cond1` and `cond2`.
                let pre = |x: u32| x.wrapping_sub(1);
                let mut run = |cond1, cond2, p1, p2, p3| {
                    if cond1 && cond2 && available[p1] && available[p2] && available[p3] {
                        q.push_back((p1, (x1, y1)));
                        q.push_back((p2, (x1, y1)));
                        q.push_back((p3, (x1, y1)));
                        src_y_history.push([p1, p2, p3]);
                    }
                };

                run(x1 > 0, y1 > 0, (pre(x1), y1), (pre(x1), pre(y1)), (x1, pre(y1)));
                run(x1 > 0, y1 + 1 < height, (pre(x1), y1), (pre(x1), y1 + 1), (x1, y1 + 1));
                run(x1 + 1 < width, y1 > 0, (x1 + 1, y1), (x1 + 1, pre(y1)), (x1, pre(y1)));
                run(x1 + 1 < width, y1 + 1 < height, (x1 + 1, y1), (x1 + 1, y1 + 1), (x1, y1 + 1));
            }
        }

        let mut destinations = vec![];
        match p2 {
            Pauli::I => {
                unreachable!("path_between: Pauli::I");
            }
            Pauli::X => {
                if x2 > 0 && available[(x2 - 1, y2)] {
                    destinations.push((x2 - 1, y2));
                }
                if x2 + 1 < width && available[(x2 + 1, y2)] {
                    destinations.push((x2 + 1, y2));
                }
            }
            Pauli::Z => {
                if y2 > 0 && available[(x2, y2 - 1)] {
                    destinations.push((x2, y2 - 1));
                }
                if y2 + 1 < height && available[(x2, y2 + 1)] {
                    destinations.push((x2, y2 + 1));
                }
            }
            Pauli::Y => {
                // We use this to avoid integer underflow. In any case, a value with underflow will
                // not be used because it will be filtered out by `cond1` and `cond2`.
                let pre = |x: u32| x.wrapping_sub(1);
                let mut run = |cond1, cond2, p1, p2, p3| {
                    if cond1 && cond2 && available[p1] && available[p2] && available[p3] {
                        destinations.push(p1);
                        destinations.push(p2);
                        destinations.push(p3);
                        dest_y_history.push([p1, p2, p3]);
                    }
                };
                run(x2 > 0, y2 > 0, (pre(x2), y2), (pre(x2), pre(y2)), (x2, pre(y2)));
                run(x2 > 0, y2 + 1 < height, (pre(x2), y2), (pre(x2), y2 + 1), (x2, y2 + 1));
                run(x2 + 1 < width, y2 > 0, (x2 + 1, y2), (x2 + 1, pre(y2)), (x2, pre(y2)));
                run(x2 + 1 < width, y2 + 1 < height, (x2 + 1, y2), (x2 + 1, y2 + 1), (x2, y2 + 1));
            }
        }

        while let Some(((x, y), (px, py))) = q.pop_front() {
            if history[(x, y)].is_some() {
                continue;
            }
            history[(x, y)] = Some((px, py));

            if destinations.contains(&(x, y)) {
                let mut path = vec![(x2, y2)];

                if p2 == Pauli::Y {
                    for region in &dest_y_history {
                        if region.contains(&(x, y)) {
                            path.extend(region.iter());
                            break;
                        }
                    }
                } else {
                    path.push((x, y));
                }

                let mut current = (x, y);
                while current != (x1, y1) {
                    let (px, py) = history[current].unwrap();
                    path.push((px, py));
                    current = (px, py);
                }

                if p1 == Pauli::Y {
                    // Modify the path so that it contains all the ocuppied ancilla qubits.
                    path.remove(path.len() - 1);
                    let s = path[path.len() - 1];
                    path.remove(path.len() - 1);

                    for region in &src_y_history {
                        if region.contains(&s) {
                            path.extend(region.iter());
                            break;
                        }
                    }
                    path.push((x1, y1));
                }

                path.reverse();
                return Some(path);
            }

            if x > 0 && available[(x - 1, y)] {
                q.push_back(((x - 1, y), (x, y)));
            }
            if x + 1 < width && available[(x + 1, y)] {
                q.push_back(((x + 1, y), (x, y)));
            }
            if y > 0 && available[(x, y - 1)] {
                q.push_back(((x, y - 1), (x, y)));
            }
            if y + 1 < height && available[(x, y + 1)] {
                q.push_back(((x, y + 1), (x, y)));
            }
        }

        None
    }

    fn schedule_pi_over_8_rotation(&mut self, rotation: &PauliRotation) -> bool {
        let support_size = rotation.axis.iter().filter(|a| **a != Pauli::I).count();
        if support_size == 1 {
            unimplemented!("schedule_pi_over_4_rotation: support size = 1");
        } else {
            unimplemented!("schedule_pi_over_4_rotation: support size > 1");
        }
    }

    fn ensure_board_occupancy(&mut self, cycle: u32) {
        let size = (self.width * self.height) as usize;
        assert_eq!(self.occupancy.len() % size, 0);
        let current_max_cycle = (self.occupancy.len() / size) as u32;

        if cycle < current_max_cycle {
            // There is nothing to do.
            return;
        }

        let new_size = ((cycle + 1) as usize) * size;
        self.occupancy.resize(new_size, BoardOccupancy::Vacant);
        for c in current_max_cycle..cycle + 1 {
            for (_q, (x, y)) in &self.data_qubit_mapping {
                let o = BoardOccupancy::IdleDataQubit;
                // We don't use set_occupancy here in order to avoid Rust borrow checker complaints.
                self.occupancy[(c * self.width * self.height + y * self.width + x) as usize] = o;
            }
        }
    }

    fn get_occupancy(&self, x: u32, y: u32, cycle: u32) -> BoardOccupancy {
        assert!(x < self.width);
        assert!(y < self.height);
        let size = (self.width * self.height) as usize;
        assert_eq!(self.occupancy.len() % size, 0);
        assert!(cycle < (self.occupancy.len() / size) as u32);

        self.occupancy[(cycle * self.width * self.height + y * self.width + x) as usize]
    }

    // Returns true if any operation is scheduled for `qubit` at or after `cycle`.
    fn has_schedule_at_or_after(&self, qubit: Qubit, cycle: u32) -> bool {
        if qubit.qubit >= self.cycle_after_last_operation_at.len() {
            return false;
        }
        self.cycle_after_last_operation_at[qubit.qubit] > cycle
    }

    // Returns true if the qubit at (x, y) is `occupancy` at all the cycles in the range.
    fn is_occupancy(
        &self,
        x: u32,
        y: u32,
        mut cycle_range: Range<u32>,
        occupancy: BoardOccupancy,
    ) -> bool {
        cycle_range.all(|c| self.get_occupancy(x, y, c) == occupancy)
    }

    // Returns true if the qubit at (x, y) is vacant at all the cycles in the range.
    fn is_vacant(&self, x: u32, y: u32, cycle_range: Range<u32>) -> bool {
        self.is_occupancy(x, y, cycle_range, BoardOccupancy::Vacant)
    }

    fn set_occupancy(&mut self, x: u32, y: u32, cycle: u32, occupancy: BoardOccupancy) {
        use BoardOccupancy::*;
        assert!(x < self.width);
        assert!(y < self.height);
        let size = (self.width * self.height) as usize;
        assert_eq!(self.occupancy.len() % size, 0);
        assert!(cycle < (self.occupancy.len() / size) as u32);
        let index = (cycle * self.width * self.height + y * self.width + x) as usize;
        assert!(self.occupancy[index] == Vacant || self.occupancy[index] == IdleDataQubit);

        self.occupancy[index] = occupancy;
    }

    fn set_cycle_after_last_operation_at(&mut self, qubit: Qubit, cycle: u32) {
        if qubit.qubit >= self.cycle_after_last_operation_at.len() {
            self.cycle_after_last_operation_at
                .resize(qubit.qubit + 1, 0);
        }
        self.cycle_after_last_operation_at[qubit.qubit] = cycle;
    }
}

impl<T: Clone + Default> Map2D<T> {
    fn new_with_default(width: u32, height: u32) -> Self {
        Map2D {
            width,
            height,
            map: vec![Default::default(); (width * height) as usize],
        }
    }
}

impl<T: Clone + Default> Index<(u32, u32)> for Map2D<T> {
    type Output = T;

    fn index(&self, index: (u32, u32)) -> &Self::Output {
        let (x, y) = index;
        assert!(x < self.width);
        assert!(y < self.height);
        &self.map[(y * self.width + x) as usize]
    }
}

impl<T: Clone + Default> IndexMut<(u32, u32)> for Map2D<T> {
    fn index_mut(&mut self, index: (u32, u32)) -> &mut Self::Output {
        let (x, y) = index;
        assert!(x < self.width);
        assert!(y < self.height);
        &mut self.map[(y * self.width + x) as usize]
    }
}

impl AncillaAvailability {
    fn new(board: &Board, cycle_range: Range<u32>) -> Self {
        let size = (board.width * board.height) as usize;
        let mut map = AncillaAvailability {
            map: Map2D {
                width: board.width,
                height: board.height,
                map: vec![false; size],
            },
        };

        for y in 0..board.height {
            for x in 0..board.width {
                map[(x, y)] = board.is_vacant(x, y, cycle_range.clone());
            }
        }

        map
    }

    fn width(&self) -> u32 {
        self.map.width
    }

    fn height(&self) -> u32 {
        self.map.height
    }
}

impl Index<(u32, u32)> for AncillaAvailability {
    type Output = bool;

    fn index(&self, index: (u32, u32)) -> &Self::Output {
        &self.map[index]
    }
}

impl IndexMut<(u32, u32)> for AncillaAvailability {
    fn index_mut(&mut self, index: (u32, u32)) -> &mut Self::Output {
        &mut self.map[index]
    }
}

#[cfg(test)]
mod tests {
    use clap::Id;

    use crate::{mapping::DataQubitMapping, pbc::Axis};

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
    fn test_init_board() {
        let mut mapping = DataQubitMapping::new(3, 4);
        mapping.map(Qubit::new(6), 0, 0);
        mapping.map(Qubit::new(7), 2, 2);

        let board = Board::new(mapping, 5);

        assert_eq!(board.width(), 3);
        assert_eq!(board.height(), 4);
        assert_eq!(
            board.data_qubit_mapping,
            HashMap::from([(Qubit::new(6), (0, 0)), (Qubit::new(7), (2, 2))])
        );
    }

    #[test]
    fn test_ensure_board_occupancy() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 4);
        mapping.map(Qubit::new(6), 0, 0);
        mapping.map(Qubit::new(7), 2, 2);
        let mut board = Board::new(mapping, 5);

        board.ensure_board_occupancy(0);
        assert_eq!(board.occupancy.len(), 3 * 4);

        board.ensure_board_occupancy(5);
        assert_eq!(board.occupancy.len(), 3 * 4 * 6);

        for cycle in 0..6 {
            assert_eq!(board.get_occupancy(0, 0, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(0, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 2, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(2, 3, cycle), Vacant);
        }
    }

    #[test]
    fn test_get_set_occupancy() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 4);
        mapping.map(Qubit::new(6), 0, 0);
        mapping.map(Qubit::new(7), 2, 2);
        let mut board = Board::new(mapping, 5);

        board.ensure_board_occupancy(5);
        for cycle in 0..6 {
            assert_eq!(board.get_occupancy(0, 0, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(0, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 2, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(2, 3, cycle), Vacant);
        }

        let id = board.issue_operation_id();
        board.set_occupancy(2, 3, 1, LatticeSurgery(id));
        let id = board.issue_operation_id();
        board.set_occupancy(2, 2, 3, DataQubitInOperation(id));
        board.set_occupancy(2, 3, 3, LatticeSurgery(id));

        for cycle in 0..6 {
            assert_eq!(board.get_occupancy(0, 0, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(0, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 1, cycle), Vacant);

            if cycle == 1 {
                assert_eq!(board.get_occupancy(2, 2, cycle), IdleDataQubit);
                assert_eq!(board.get_occupancy(2, 3, cycle), LatticeSurgery(OperationId { id: 0 }));
            } else if cycle == 3 {
                assert_eq!(
                    board.get_occupancy(2, 2, cycle),
                    DataQubitInOperation(OperationId { id: 1 })
                );
                assert_eq!(board.get_occupancy(2, 3, cycle), LatticeSurgery(OperationId { id: 1 }));
            } else {
                assert_eq!(board.get_occupancy(2, 2, cycle), IdleDataQubit);
                assert_eq!(board.get_occupancy(2, 3, cycle), Vacant);
            }
        }

        assert!(board.is_vacant(0, 0, 0..0));
        assert!(!board.is_vacant(0, 0, 0..1));
        assert!(!board.is_vacant(0, 0, 0..2));
        assert!(!board.is_vacant(0, 0, 1..2));
        assert!(board.is_vacant(2, 3, 0..1));
        assert!(!board.is_vacant(2, 3, 0..2));
        assert!(!board.is_vacant(2, 3, 1..2));
        assert!(!board.is_vacant(2, 3, 1..3));
        assert!(board.is_vacant(2, 3, 2..3));
        assert!(board.is_vacant(1, 3, 0..6));
    }

    #[test]
    fn test_schedule_pauli_rotation() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 4);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 2, 2);
        let mut board = Board::new(mapping, 5);
        board.ensure_board_occupancy(6);

        for cycle in 0..6 {
            assert_eq!(board.get_occupancy(0, 0, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(0, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 1, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(1, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 2, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(2, 3, cycle), Vacant);
        }

        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::Zero,
            axis: new_axis("IXX")
        }));

        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver2,
            axis: new_axis("YZY")
        }));

        for cycle in 0..6 {
            assert_eq!(board.get_occupancy(0, 0, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(0, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(0, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 1, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(1, 2, cycle), Vacant);
            assert_eq!(board.get_occupancy(1, 3, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 0, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 1, cycle), Vacant);
            assert_eq!(board.get_occupancy(2, 2, cycle), IdleDataQubit);
            assert_eq!(board.get_occupancy(2, 3, cycle), Vacant);
        }
    }

    #[test]
    fn test_schedule_single_qubit_clifford_rotation_with_x_axis() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 3);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 2, 2);
        let mut board = Board::new(mapping, 5);
        board.ensure_board_occupancy(6);

        let id = board.issue_operation_id();
        board.set_occupancy(0, 0, 0, DataQubitInOperation(id));
        board.set_cycle_after_last_operation_at(Qubit::new(0), 1);

        let id = board.issue_operation_id();
        board.set_occupancy(1, 0, 1, LatticeSurgery(id));

        // Rejected because the qubit is already in operation at cycle 0.
        assert!(!board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XII")
        })));

        board.cycle = 1;
        // Rejected because the ancilla qubit at (1, 0) is already used.
        assert!(!board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XII")
        })));

        assert!(board.is_occupancy(0, 0, 0..1, DataQubitInOperation(OperationId { id: 0 })));
        assert!(board.is_occupancy(0, 0, 1..6, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 0..6, Vacant));
        assert!(board.is_occupancy(0, 2, 0..6, Vacant));
        assert!(board.is_occupancy(1, 0, 0..1, Vacant));
        assert!(board.is_occupancy(1, 0, 1..2, LatticeSurgery(OperationId { id: 1 })));
        assert!(board.is_occupancy(1, 0, 2..6, Vacant));
        assert!(board.is_occupancy(1, 1, 1..6, IdleDataQubit));
        assert!(board.is_occupancy(1, 2, 0..6, Vacant));
        assert!(board.is_occupancy(2, 0, 0..6, Vacant));
        assert!(board.is_occupancy(2, 1, 0..6, Vacant));
        assert!(board.is_occupancy(2, 2, 0..6, IdleDataQubit));

        board.cycle = 2;
        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XII")
        }));
        let id = OperationId { id: 3 };

        assert_eq!(board.occupancy.len(), 3 * 3 * 17);

        assert!(board.is_occupancy(0, 0, 0..1, DataQubitInOperation(OperationId { id: 0 })));
        assert!(board.is_occupancy(0, 0, 1..2, IdleDataQubit));
        assert!(board.is_occupancy(0, 0, 2..7, DataQubitInOperation(id)));
        assert!(board.is_occupancy(0, 0, 7..17, IdleDataQubit));
        assert!(board.is_occupancy(1, 0, 0..1, Vacant));
        assert!(board.is_occupancy(1, 0, 1..2, LatticeSurgery(OperationId { id: 1 })));
        assert!(board.is_occupancy(1, 0, 2..7, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 0, 7..11, YMeasurement));
        assert!(board.is_occupancy(1, 0, 11..17, Vacant));
        assert!(board.is_occupancy(0, 1, 0..17, Vacant));
        assert!(board.is_occupancy(0, 2, 0..17, Vacant));
        assert!(board.is_occupancy(1, 1, 0..17, IdleDataQubit));
        assert!(board.is_occupancy(1, 2, 0..17, Vacant));
        assert!(board.is_occupancy(2, 0, 0..17, Vacant));
        assert!(board.is_occupancy(2, 1, 0..17, Vacant));
        assert!(board.is_occupancy(2, 2, 0..17, IdleDataQubit));

        assert!(board.has_schedule_at_or_after(Qubit::new(0), 6));
        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 7));
        assert!(!board.has_schedule_at_or_after(Qubit::new(1), 0));
        assert!(!board.has_schedule_at_or_after(Qubit::new(2), 0));
    }

    #[test]
    fn test_schedule_single_qubit_clifford_rotation_with_z_axis() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 3);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 2, 2);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(5);

        let id = board.issue_operation_id();
        board.set_occupancy(2, 2, 0, DataQubitInOperation(id));
        board.set_cycle_after_last_operation_at(Qubit::new(2), 1);

        // Rejected because the qubit is already in operation at cycle 0.
        assert!(!board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("IIZ")
        }));

        assert!(board.is_occupancy(0, 0, 0..5, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 0..5, Vacant));
        assert!(board.is_occupancy(0, 2, 0..5, Vacant));
        assert!(board.is_occupancy(1, 0, 0..5, Vacant));
        assert!(board.is_occupancy(1, 1, 0..5, IdleDataQubit));
        assert!(board.is_occupancy(1, 2, 0..5, Vacant));
        assert!(board.is_occupancy(2, 0, 0..5, Vacant));
        assert!(board.is_occupancy(2, 1, 0..5, Vacant));
        assert!(board.is_occupancy(2, 2, 0..1, DataQubitInOperation(OperationId { id: 0 })));
        assert!(board.is_occupancy(2, 2, 1..5, IdleDataQubit));

        board.cycle = 6;
        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("IIZ")
        }));

        let id = OperationId { id: 1 };
        assert_eq!(board.occupancy.len(), 3 * 3 * 16);

        assert!(board.is_occupancy(0, 0, 0..16, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 0..16, Vacant));
        assert!(board.is_occupancy(0, 2, 0..16, Vacant));
        assert!(board.is_occupancy(1, 0, 0..16, Vacant));
        assert!(board.is_occupancy(1, 1, 0..16, IdleDataQubit));
        assert!(board.is_occupancy(1, 2, 0..16, Vacant));
        assert!(board.is_occupancy(2, 0, 0..16, Vacant));
        assert!(board.is_occupancy(2, 1, 0..6, YInitialization));
        assert!(board.is_occupancy(2, 1, 6..9, LatticeSurgery(id)));
        assert!(board.is_occupancy(2, 1, 9..16, Vacant));
        assert!(board.is_occupancy(2, 2, 0..1, DataQubitInOperation(OperationId { id: 0 })));
        assert!(board.is_occupancy(2, 2, 1..6, IdleDataQubit));
        assert!(board.is_occupancy(2, 2, 6..9, DataQubitInOperation(id)));
        assert!(board.is_occupancy(2, 2, 9..16, IdleDataQubit));

        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 0));
        assert!(!board.has_schedule_at_or_after(Qubit::new(1), 0));
        assert!(board.has_schedule_at_or_after(Qubit::new(2), 8));
        assert!(!board.has_schedule_at_or_after(Qubit::new(2), 9));
    }

    #[test]
    fn test_schedule_single_qubit_clifford_rotation_with_z_axis_competing_neighbors() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 3);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 2, 1);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(0);

        let id = board.issue_operation_id();
        board.set_occupancy(0, 0, 0, LatticeSurgery(id));
        board.set_occupancy(1, 2, 0, LatticeSurgery(id));

        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("ZII")
        }));
        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("IZI")
        }));
        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("IIZ")
        }));

        assert_eq!(board.occupancy.len(), 3 * 3 * 10);

        assert!(board.is_occupancy(0, 0, 0..1, LatticeSurgery(OperationId { id: 0 })));
        assert!(board.is_occupancy(0, 0, 1..10, Vacant));
        assert!(board.is_occupancy(0, 1, 0..3, DataQubitInOperation(OperationId { id: 1 })));
        assert!(board.is_occupancy(0, 1, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(0, 2, 0..3, LatticeSurgery(OperationId { id: 1 })));
        assert!(board.is_occupancy(0, 2, 3..6, YMeasurement));
        assert!(board.is_occupancy(0, 2, 6..10, Vacant));
        assert!(board.is_occupancy(1, 0, 0..3, LatticeSurgery(OperationId { id: 2 })));
        assert!(board.is_occupancy(1, 0, 3..6, YMeasurement));
        assert!(board.is_occupancy(1, 0, 6..10, Vacant));
        assert!(board.is_occupancy(1, 1, 0..3, DataQubitInOperation(OperationId { id: 2 })));
        assert!(board.is_occupancy(1, 1, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(1, 2, 0..1, LatticeSurgery(OperationId { id: 0 })));
        assert!(board.is_occupancy(1, 2, 1..10, Vacant));
        assert!(board.is_occupancy(2, 0, 0..3, LatticeSurgery(OperationId { id: 3 })));
        assert!(board.is_occupancy(2, 0, 3..6, YMeasurement));
        assert!(board.is_occupancy(2, 0, 6..10, Vacant));
        assert!(board.is_occupancy(2, 1, 0..3, DataQubitInOperation(OperationId { id: 3 })));
        assert!(board.is_occupancy(2, 1, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(2, 2, 0..10, Vacant));

        assert!(board.has_schedule_at_or_after(Qubit::new(0), 2));
        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 3));
        assert!(board.has_schedule_at_or_after(Qubit::new(1), 2));
        assert!(!board.has_schedule_at_or_after(Qubit::new(1), 3));
        assert!(board.has_schedule_at_or_after(Qubit::new(2), 2));
        assert!(!board.has_schedule_at_or_after(Qubit::new(2), 3));
    }

    #[test]
    fn test_schedule_single_qubit_clifford_rotation_with_x_axis_competing_neighbors() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 3);
        mapping.map(Qubit::new(0), 1, 0);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 1, 2);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(0);

        let id = board.issue_operation_id();
        board.set_occupancy(0, 0, 0, LatticeSurgery(id));
        let id = board.issue_operation_id();
        board.set_occupancy(2, 1, 0, LatticeSurgery(id));

        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XII")
        }));
        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("IXI")
        }));
        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("IIX")
        }));

        assert_eq!(board.occupancy.len(), 3 * 3 * 10);

        assert!(board.is_occupancy(0, 0, 0..1, LatticeSurgery(OperationId { id: 0 })));
        assert!(board.is_occupancy(0, 0, 1..10, Vacant));
        assert!(board.is_occupancy(0, 1, 0..3, LatticeSurgery(OperationId { id: 3 })));
        assert!(board.is_occupancy(0, 1, 3..6, YMeasurement));
        assert!(board.is_occupancy(0, 1, 6..10, Vacant));
        assert!(board.is_occupancy(0, 2, 0..3, LatticeSurgery(OperationId { id: 4 })));
        assert!(board.is_occupancy(0, 2, 3..6, YMeasurement));
        assert!(board.is_occupancy(0, 2, 6..10, Vacant));
        assert!(board.is_occupancy(1, 0, 0..3, DataQubitInOperation(OperationId { id: 2 })));
        assert!(board.is_occupancy(1, 0, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(1, 1, 0..3, DataQubitInOperation(OperationId { id: 3 })));
        assert!(board.is_occupancy(1, 1, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(1, 2, 0..3, DataQubitInOperation(OperationId { id: 4 })));
        assert!(board.is_occupancy(1, 2, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(2, 0, 0..3, LatticeSurgery(OperationId { id: 2 })));
        assert!(board.is_occupancy(2, 0, 3..6, YMeasurement));
        assert!(board.is_occupancy(2, 0, 6..10, Vacant));
        assert!(board.is_occupancy(2, 1, 0..1, LatticeSurgery(OperationId { id: 1 })));
        assert!(board.is_occupancy(2, 1, 1..10, Vacant));
        assert!(board.is_occupancy(2, 2, 0..10, Vacant));

        assert!(board.has_schedule_at_or_after(Qubit::new(0), 2));
        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 3));
        assert!(board.has_schedule_at_or_after(Qubit::new(1), 2));
        assert!(!board.has_schedule_at_or_after(Qubit::new(1), 3));
        assert!(board.has_schedule_at_or_after(Qubit::new(2), 2));
        assert!(!board.has_schedule_at_or_after(Qubit::new(2), 3));
    }

    #[test]
    fn test_schedule_two_qubit_clifford_rotation_conflicting() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(4, 3);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 3, 2);
        let mut board = Board::new(mapping, 3);

        board.ensure_board_occupancy(12);
        let id = board.issue_operation_id();
        board.set_occupancy(0, 0, 0, DataQubitInOperation(id));
        board.set_occupancy(3, 2, 11, DataQubitInOperation(id));
        board.set_cycle_after_last_operation_at(Qubit::new(0), 1);
        board.set_cycle_after_last_operation_at(Qubit::new(2), 11);

        assert!(!board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XXI")
        }));
        assert!(!board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("IXX")
        }));
    }

    #[test]
    fn test_path_between() {
        use Pauli::*;
        #[rustfmt::skip]
        let graphical_map: [&str; 8] = [
            "X       ",
            " X X X X",
            " X      ",
            " X X XXX",
            "X  X   X",
            " X X X X",
            "   X    ",
            "XXXX X X",
        ];
        let mut map = AncillaAvailability {
            map: Map2D {
                width: 8,
                height: 8,
                map: vec![true; 64],
            },
        };
        for y in 0..8_usize {
            for x in 0..8_usize {
                map[(x as u32, y as u32)] = graphical_map[y].chars().nth(x) == Some(' ');
            }
        }

        assert!(Board::path_between((1, 7, Z), (3, 7, Z), &map).is_none());
        assert!(Board::path_between((1, 7, X), (3, 7, X), &map).is_none());
        assert!(Board::path_between((1, 7, X), (1, 1, X), &map).is_none());
        assert!(Board::path_between((1, 7, Y), (1, 1, X), &map).is_none());
        assert!(Board::path_between((3, 5, Y), (5, 5, X), &map).is_none());
        assert!(Board::path_between((1, 1, X), (1, 7, Y), &map).is_none());
        assert!(Board::path_between((5, 5, X), (3, 5, Y), &map).is_none());

        assert_eq!(
            Board::path_between((1, 7, Z), (1, 3, X), &map),
            Some(vec![(1, 7), (1, 6), (2, 6), (2, 5), (2, 4), (2, 3), (1, 3)])
        );
        assert_eq!(
            Board::path_between((1, 1, X), (7, 7, Z), &map),
            Some(vec![
                (1, 1),
                (2, 1),
                (2, 2),
                (3, 2),
                (4, 2),
                (4, 3),
                (4, 4),
                (5, 4),
                (6, 4),
                (6, 5),
                (6, 6),
                (7, 6),
                (7, 7)
            ])
        );
        assert_eq!(
            Board::path_between((1, 1, Z), (7, 5, Z), &map),
            Some(vec![
                (1, 1),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3),
                (4, 4),
                (5, 4),
                (6, 4),
                (6, 5),
                (6, 6),
                (7, 6),
                (7, 5)
            ])
        );
        assert_eq!(
            Board::path_between((1, 1, Y), (3, 1, X), &map),
            Some(vec![(1, 1), (1, 0), (2, 0), (2, 1), (3, 1)])
        );
        assert_eq!(
            Board::path_between((3, 1, Y), (5, 1, X), &map),
            Some(vec![(3, 1), (3, 0), (4, 0), (4, 1), (5, 1)])
        );
        assert_eq!(
            Board::path_between((3, 1, Y), (3, 3, Z), &map),
            Some(vec![(3, 1), (3, 2), (2, 2), (2, 1), (3, 3)])
        );
        assert_eq!(
            Board::path_between((3, 1, Y), (5, 3, Z), &map),
            Some(vec![(3, 1), (3, 2), (4, 2), (4, 1), (5, 2), (5, 3)])
        );
        assert_eq!(
            Board::path_between((5, 3, Y), (3, 1, X), &map),
            Some(vec![(5, 3), (5, 2), (4, 2), (4, 3), (4, 1), (3, 1)])
        );
        assert_eq!(
            Board::path_between((3, 1, X), (1, 1, Y), &map),
            Some(vec![(3, 1), (1, 0), (2, 0), (2, 1), (1, 1)])
        );
        assert_eq!(
            Board::path_between((5, 1, X), (3, 1, Y), &map),
            Some(vec![(5, 1), (3, 0), (4, 0), (4, 1), (3, 1)])
        );
        assert_eq!(
            Board::path_between((3, 3, Z), (3, 1, Y), &map),
            Some(vec![(3, 3), (3, 2), (2, 2), (2, 1), (3, 1)])
        );
        assert_eq!(
            Board::path_between((5, 3, Z), (3, 1, Y), &map),
            Some(vec![(5, 3), (5, 2), (3, 2), (4, 2), (4, 1), (3, 1)])
        );
        assert_eq!(
            Board::path_between((3, 1, X), (5, 3, Y), &map),
            Some(vec![(3, 1), (4, 1), (5, 2), (4, 2), (4, 3), (5, 3)])
        );
    }

    #[test]
    fn test_schedule_two_qubit_clifford_rotation_data_qubit_is_busy() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(2, 2);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(0);
        let id = board.issue_operation_id();
        board.set_occupancy(0, 0, 0, DataQubitInOperation(id));
        board.set_cycle_after_last_operation_at(Qubit::new(0), 1);

        assert!(!board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XX")
        })));
    }

    #[test]
    fn test_schedule_two_qubit_clifford_rotation_unreachable() {
        let mut mapping = DataQubitMapping::new(2, 2);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        let mut board = Board::new(mapping, 3);
        assert!(!board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XX")
        })));
    }

    // Check if reachability is checked with `cycle..cycle + distance`.
    #[test]
    fn test_schedule_two_qubit_clifford_rotation_unreachable_2() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 3);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 2, 2);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(3);
        let id = board.issue_operation_id();
        board.set_occupancy(1, 0, 2, LatticeSurgery(id));
        assert!(!board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XX")
        })));
    }

    // There is no space around the path between the data qubits.
    // The only qubit in the path is not available, because the path is not linear at the qubit.
    #[test]
    fn test_schedule_two_qubit_clifford_rotation_no_space_for_y_eigenstate_1() {
        let mut mapping = DataQubitMapping::new(2, 2);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(3);

        assert!(!board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("ZX")
        })));
    }

    // There is no space around the path between the data qubits.
    // The only qubit in the path is not available, because there is another lattice surgery.
    #[test]
    fn test_schedule_two_qubit_clifford_rotation_no_space_for_y_eigenstate_2() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 1);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 2, 0);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(3);
        let id = board.issue_operation_id();
        board.set_occupancy(1, 0, 3, LatticeSurgery(id));

        assert!(!board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XX")
        })));
    }

    // Both the inplace and adjacent options are forbidden by another lattice surgery.
    #[test]
    fn test_schedule_two_qubit_clifford_rotation_no_space_for_y_eigenstate_3() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(2, 3);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 0, 2);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(3);
        let id = board.issue_operation_id();
        board.set_occupancy(0, 1, 3, LatticeSurgery(id));
        board.set_occupancy(1, 1, 3, LatticeSurgery(id));

        assert!(!board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("ZZ")
        })));
    }

    #[test]
    fn test_schedule_two_qubit_clifford_rotation_no_space_for_y_eigenstate_4() {
        let mut mapping = DataQubitMapping::new(2, 4);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 0, 2);
        let mut board = Board::new(mapping, 3);

        assert!(!board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("YY")
        })));
    }

    #[test]
    fn test_schedule_xz_pi_over_4_rotation_with_y_initialization() {
        use BoardOccupancy::*;
        let width = 2_u32;
        let height = 4_u32;
        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 3);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(11);
        board.set_occupancy(1, 1, 3, YMeasurement);
        board.set_occupancy(1, 2, 4, YMeasurement);
        board.set_occupancy(1, 1, 11, YInitialization);
        let id = OperationId { id: 0 };

        board.cycle = 8;
        assert!(board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XZ")
        })));

        assert!(board.is_occupancy(0, 0, 0..8, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 0..2, Vacant));
        assert!(board.is_occupancy(0, 1, 2..8, YInitialization));
        assert!(board.is_occupancy(0, 2, 0..8, Vacant));
        assert!(board.is_occupancy(0, 3, 0..8, Vacant));
        assert!(board.is_occupancy(1, 0, 0..8, Vacant));
        assert!(board.is_occupancy(1, 1, 0..3, Vacant));
        assert!(board.is_occupancy(1, 1, 3..4, YMeasurement));
        assert!(board.is_occupancy(1, 1, 4..8, Vacant));
        assert!(board.is_occupancy(1, 2, 0..4, Vacant));
        assert!(board.is_occupancy(1, 2, 4..5, YMeasurement));
        assert!(board.is_occupancy(1, 2, 5..8, Vacant));
        assert!(board.is_occupancy(1, 3, 0..8, IdleDataQubit));

        assert!(board.is_occupancy(0, 0, 8..11, DataQubitInOperation(id)));
        assert!(board.is_occupancy(0, 1, 8..11, LatticeSurgery(id)));
        assert!(board.is_occupancy(0, 2, 8..11, Vacant));
        assert!(board.is_occupancy(0, 3, 8..11, Vacant));
        assert!(board.is_occupancy(1, 0, 8..11, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 1, 8..11, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 2, 8..11, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 3, 8..11, DataQubitInOperation(id)));

        assert!(board.is_occupancy(0, 0, 11..12, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 11..12, Vacant));
        assert!(board.is_occupancy(0, 2, 11..12, Vacant));
        assert!(board.is_occupancy(0, 3, 11..12, Vacant));
        assert!(board.is_occupancy(1, 0, 11..12, Vacant));
        assert!(board.is_occupancy(1, 1, 11..12, YInitialization));
        assert!(board.is_occupancy(1, 2, 11..12, Vacant));
        assert!(board.is_occupancy(1, 3, 11..12, IdleDataQubit));
    }

    #[test]
    fn test_schedule_xz_pi_over_4_rotation_with_adjacent_y_measurement() {
        use BoardOccupancy::*;
        let width = 2_u32;
        let height = 4_u32;
        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 3);
        let mut board = Board::new(mapping, 3);
        board.ensure_board_occupancy(4);
        board.set_occupancy(1, 1, 3, YInitialization);
        board.set_occupancy(1, 2, 4, YInitialization);
        let id = OperationId { id: 0 };

        assert!(board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XZ")
        })));

        assert!(board.is_occupancy(0, 0, 0..3, DataQubitInOperation(id)));
        assert!(board.is_occupancy(0, 1, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(0, 2, 0..3, Vacant));
        assert!(board.is_occupancy(0, 3, 0..3, Vacant));
        assert!(board.is_occupancy(1, 0, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 1, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 2, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 3, 0..3, DataQubitInOperation(id)));

        assert!(board.is_occupancy(0, 0, 3..6, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 3..6, YMeasurement));
        assert!(board.is_occupancy(0, 2, 3..6, Vacant));
        assert!(board.is_occupancy(0, 3, 3..6, Vacant));
        assert!(board.is_occupancy(1, 0, 3..6, Vacant));
        assert!(board.is_occupancy(1, 1, 3..4, YInitialization));
        assert!(board.is_occupancy(1, 1, 4..6, Vacant));
        assert!(board.is_occupancy(1, 2, 3..4, Vacant));
        assert!(board.is_occupancy(1, 2, 4..5, YInitialization));
        assert!(board.is_occupancy(1, 2, 5..6, Vacant));
        assert!(board.is_occupancy(1, 3, 3..6, IdleDataQubit));

        assert!(board.is_occupancy(0, 0, 6..7, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 6..7, Vacant));
        assert!(board.is_occupancy(0, 2, 6..7, Vacant));
        assert!(board.is_occupancy(0, 3, 6..7, Vacant));
        assert!(board.is_occupancy(1, 0, 6..7, Vacant));
        assert!(board.is_occupancy(1, 1, 6..7, Vacant));
        assert!(board.is_occupancy(1, 2, 6..7, Vacant));
        assert!(board.is_occupancy(1, 3, 6..7, IdleDataQubit));

        assert!(board.has_schedule_at_or_after(Qubit::new(0), 2));
        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 3));
        assert!(board.has_schedule_at_or_after(Qubit::new(1), 2));
        assert!(!board.has_schedule_at_or_after(Qubit::new(1), 3));
    }

    #[test]
    fn test_schedule_yy_pi_over_4_rotation_with_adjacent_y_measurement() {
        use BoardOccupancy::*;
        let width = 3_u32;
        let height = 4_u32;
        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 0, 2);
        let mut board = Board::new(mapping, 3);

        assert!(board.schedule(&Operator::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("YY")
        })));

        let id = OperationId { id: 0 };

        assert!(board.is_occupancy(0, 0, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(0, 1, 0..3, DataQubitInOperation(id)));
        assert!(board.is_occupancy(0, 2, 0..3, DataQubitInOperation(id)));
        assert!(board.is_occupancy(0, 3, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 0, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 1, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 2, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 3, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(2, 0, 0..3, LatticeSurgery(id)));
        assert!(board.is_occupancy(2, 1, 0..3, Vacant));
        assert!(board.is_occupancy(2, 2, 0..3, Vacant));
        assert!(board.is_occupancy(2, 3, 0..3, Vacant));

        assert!(board.is_occupancy(0, 0, 3..7, Vacant));
        assert!(board.is_occupancy(0, 1, 3..7, IdleDataQubit));
        assert!(board.is_occupancy(0, 2, 3..7, IdleDataQubit));
        assert!(board.is_occupancy(0, 3, 3..7, Vacant));
        assert!(board.is_occupancy(1, 0, 3..7, Vacant));
        assert!(board.is_occupancy(1, 1, 3..7, Vacant));
        assert!(board.is_occupancy(1, 2, 3..7, Vacant));
        assert!(board.is_occupancy(1, 3, 3..7, Vacant));
        assert!(board.is_occupancy(2, 0, 3..6, YMeasurement));
        assert!(board.is_occupancy(2, 0, 6..7, Vacant));
        assert!(board.is_occupancy(2, 1, 3..7, Vacant));
        assert!(board.is_occupancy(2, 2, 3..7, Vacant));
        assert!(board.is_occupancy(2, 3, 3..7, Vacant));
    }
}
