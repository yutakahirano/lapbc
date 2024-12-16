use std::collections::{HashMap, HashSet};
use std::ops::{Index, IndexMut};

use rand::{thread_rng, Rng};

use crate::board::Map2D;
use crate::board::OperationId;
use crate::board::{y_measurement_cost, Configuration};
use crate::board::{Board, OperationWithAdditionalData};
use crate::board::{BoardOccupancy, Position};

struct Range2D {
    width: u32,
    height: u32,
    x: u32,
    y: u32,
}

fn range_2d(width: u32, height: u32) -> Range2D {
    Range2D {
        width,
        height,
        x: 0,
        y: 0,
    }
}

impl Iterator for Range2D {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.y == self.height {
            return None;
        }

        let ret = (self.x, self.y);
        self.x += 1;
        if self.x == self.width {
            self.x = 0;
            self.y += 1;
        }
        Some(ret)
    }
}

pub struct OccupancyMap {
    width: u32,
    height: u32,
    map: Vec<BoardOccupancy>,
}

impl Index<(u32, u32, u32)> for OccupancyMap {
    type Output = BoardOccupancy;

    fn index(&self, (x, y, cycle): (u32, u32, u32)) -> &Self::Output {
        &self.map[(cycle * self.width * self.height + y * self.width + x) as usize]
    }
}

impl IndexMut<(u32, u32, u32)> for OccupancyMap {
    fn index_mut(&mut self, (x, y, cycle): (u32, u32, u32)) -> &mut Self::Output {
        &mut self.map[(cycle * self.width * self.height + y * self.width + x) as usize]
    }
}

impl OccupancyMap {
    fn new(board: &Board) -> Self {
        let width = board.width();
        let height = board.height();
        let end_cycle = board.get_last_end_cycle();

        let mut map = OccupancyMap {
            width,
            height,
            map: vec![BoardOccupancy::Vacant; (width * height * end_cycle) as usize],
        };

        for cycle in 0..end_cycle {
            for y in 0..height {
                for x in 0..width {
                    map[(x, y, cycle)] = board.get_occupancy(x, y, cycle);
                }
            }
        }

        map
    }

    fn end_cycle(&self) -> u32 {
        self.map.len() as u32 / (self.width * self.height)
    }
}

#[derive(Debug)]
enum PiOver8RotationState {
    Distillation {
        steps: Vec<(Position, u32)>,
        has_magic_state: bool,
    },
    LatticeSurgery {
        steps: u32,
    },
    Correction {
        steps: u32,
    },
}

#[derive(Debug)]
enum PiOver4RotationState {
    Initial,
    LatticeSurgery { steps: u32 },
    Correction { steps: u32 },
}

pub struct Runner {
    operations: HashMap<OperationId, OperationWithAdditionalData>,
    schedule: OccupancyMap,
    conf: Configuration,

    delay_at: Map2D<u32>,
    end_cycle_at: Map2D<u32>,
    runtime_cycle: u32,
    pi_over_4_rotation_states: HashMap<OperationId, PiOver4RotationState>,
    pi_over_8_rotation_states: HashMap<OperationId, PiOver8RotationState>,
    removed_operation_ids: HashSet<OperationId>,
}

impl Runner {
    pub fn new(board: &Board) -> Self {
        let mut end_cycle_at = Map2D::new_with_value(board.width(), board.height(), 0);
        let schedule = OccupancyMap::new(board);
        for (x, y) in range_2d(board.width(), board.height()) {
            let mut cycle = schedule.end_cycle();
            while cycle > 0 {
                if !schedule[(x, y, cycle - 1)].is_vacant_or_idle() {
                    break;
                }
                cycle -= 1;
            }
            end_cycle_at[(x, y)] = cycle;
        }
        let operations = board
            .operations()
            .iter()
            .map(|op| (op.id(), op.clone()))
            .collect::<HashMap<_, _>>();
        Runner {
            operations,
            schedule,
            conf: board.configuration().clone(),
            delay_at: Map2D::new_with_value(board.width(), board.height(), 0),
            end_cycle_at,
            runtime_cycle: 0,
            pi_over_4_rotation_states: HashMap::new(),
            pi_over_8_rotation_states: HashMap::new(),
            removed_operation_ids: HashSet::new(),
        }
    }

    fn register_initial_state(&mut self, occupancy: BoardOccupancy) {
        use OperationWithAdditionalData::*;

        if let Some(id) = occupancy.operation_id() {
            if self.removed_operation_ids.contains(&id) {
                return;
            }
            let op = &self.operations[&id];
            match op {
                PiOver4Rotation { .. } => {
                    self.pi_over_4_rotation_states
                        .entry(id)
                        .or_insert(PiOver4RotationState::Initial);
                }
                PiOver8Rotation {
                    distillation_qubits,
                    ..
                } => {
                    let steps = distillation_qubits
                        .iter()
                        .map(|pos| (*pos, 0_u32))
                        .collect::<Vec<_>>();
                    let has_magic_state = false;
                    self.pi_over_8_rotation_states.entry(id).or_insert(
                        PiOver8RotationState::Distillation {
                            steps,
                            has_magic_state,
                        },
                    );
                }
                SingleQubitPiOver8RotationBlock { .. } => unimplemented!(),
            }
        }
    }

    fn perform_pi_over_4_rotation_state_transition(&mut self) {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        let mut to_be_removed = vec![];
        for (id, state) in &mut self.pi_over_4_rotation_states {
            let op = &self.operations[id];
            let mut qubits = match op {
                PiOver4Rotation {
                    targets,
                    ancilla_qubits,
                    ..
                } => targets
                    .iter()
                    .map(|(pos, _)| pos)
                    .chain(ancilla_qubits.iter()),
                _ => unreachable!(),
            };
            loop {
                match state {
                    PiOver4RotationState::Initial => {
                        let is_ready = |pos: &Position| {
                            let cycle_on_schedule =
                                self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                            let occupancy = &self.schedule[(pos.x, pos.y, cycle_on_schedule)];
                            matches!(occupancy, LatticeSurgery(id) | DataQubitInOperation(id) if *id == op.id())
                        };
                        if qubits.all(is_ready) {
                            *state = PiOver4RotationState::LatticeSurgery { steps: 0 };
                            continue;
                        }
                    }
                    PiOver4RotationState::LatticeSurgery { steps } => {
                        if steps == &self.conf.code_distance {
                            *state = PiOver4RotationState::Correction { steps: 0 };
                            continue;
                        }
                    }
                    PiOver4RotationState::Correction { steps } => {
                        let is_correction = |pos: &Position| {
                            let cycle_on_schedule =
                                self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                            if cycle_on_schedule >= self.schedule.end_cycle() {
                                return false;
                            }
                            let occupancy = &self.schedule[(pos.x, pos.y, cycle_on_schedule)];
                            matches!(occupancy, YMeasurement(id) if *id == op.id())
                        };
                        let y_measurement_cost = y_measurement_cost(self.conf.code_distance);
                        let has_correction_occupancy = qubits.any(is_correction);
                        if has_correction_occupancy {
                            assert!(*steps < y_measurement_cost);
                        } else {
                            assert!(*steps == 0 || *steps == y_measurement_cost);
                            to_be_removed.push(*id);
                        }
                    }
                }
                break;
            }
        }
        self.removed_operation_ids.extend(to_be_removed.iter());
        for id in to_be_removed {
            self.pi_over_4_rotation_states.remove(&id);
        }
    }

    // Returns true when this state must be removed.
    fn perform_pi_over_8_rotation_state_transition(&mut self) {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        let mut to_be_removed = vec![];
        for (id, state) in &mut self.pi_over_8_rotation_states {
            loop {
                match state {
                    PiOver8RotationState::Distillation {
                        has_magic_state, ..
                    } => {
                        let op = &self.operations[id];
                        let mut qubits = match op {
                            PiOver8Rotation {
                                targets,
                                routing_qubits,
                                distillation_qubits,
                                ..
                            } => targets
                                .iter()
                                .map(|(pos, _)| pos)
                                .chain(routing_qubits.iter())
                                .chain(distillation_qubits.iter()),
                            _ => unreachable!(),
                        };

                        if !*has_magic_state {
                            break;
                        }
                        let is_ready = |pos: &Position| {
                            let cycle_on_schedule =
                                self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                            let occupancy = &self.schedule[(pos.x, pos.y, cycle_on_schedule)];
                            matches!(occupancy, LatticeSurgery(id) | DataQubitInOperation(id) | MagicStateDistillation(id) if *id == op.id())
                        };
                        if qubits.all(is_ready) {
                            *state = PiOver8RotationState::LatticeSurgery { steps: 0 };
                            continue;
                        }
                    }
                    PiOver8RotationState::LatticeSurgery { steps } => {
                        if *steps == self.conf.code_distance {
                            *state = PiOver8RotationState::Correction { steps: 0 };
                            continue;
                        }
                    }
                    PiOver8RotationState::Correction { steps } => {
                        if *steps == y_measurement_cost(self.conf.code_distance) {
                            to_be_removed.push(*id);
                        }
                    }
                }
                break;
            }
        }
        self.removed_operation_ids.extend(to_be_removed.iter());
        for id in to_be_removed {
            self.pi_over_8_rotation_states.remove(&id);
        }
    }

    fn process_pi_over_4_rotation(&mut self) {
        use BoardOccupancy::*;
        for (id, state) in &mut self.pi_over_4_rotation_states {
            match state {
                PiOver4RotationState::Initial => {
                    // We are waiting for some qubits. Add delay to the ready qubits.
                    let op = &self.operations[id];
                    let qubits = match op {
                        OperationWithAdditionalData::PiOver4Rotation {
                            targets,
                            ancilla_qubits,
                            ..
                        } => targets
                            .iter()
                            .map(|(pos, _)| pos)
                            .chain(ancilla_qubits.iter()),
                        _ => unreachable!(),
                    };
                    for pos in qubits {
                        let cycle_on_schedule = self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                        let occupancy = &self.schedule[(pos.x, pos.y, cycle_on_schedule)];
                        if matches!(occupancy, LatticeSurgery(id) | DataQubitInOperation(id) if *id == op.id())
                        {
                            self.delay_at[(pos.x, pos.y)] += 1;
                        }
                    }
                }
                PiOver4RotationState::LatticeSurgery { steps } => {
                    *steps += 1;
                }
                PiOver4RotationState::Correction { steps } => {
                    *steps += 1;
                }
            }
        }
    }

    fn process_pi_over_8_rotation<R: Rng>(&mut self, rng: &mut R) {
        use BoardOccupancy::*;
        for (id, state) in &mut self.pi_over_8_rotation_states {
            let op = &self.operations[id];
            let qubits = match op {
                OperationWithAdditionalData::PiOver8Rotation {
                    targets,
                    routing_qubits,
                    distillation_qubits,
                    ..
                } => targets
                    .iter()
                    .map(|(pos, _)| pos)
                    .chain(routing_qubits.iter())
                    .chain(distillation_qubits.iter()),
                _ => unreachable!(),
            };
            match state {
                PiOver8RotationState::Distillation {
                    steps,
                    has_magic_state,
                } => {
                    for (pos, steps) in steps {
                        let cycle_on_schedule = self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                        let occupancy = &self.schedule[(pos.x, pos.y, cycle_on_schedule)];
                        if !matches!(occupancy, LatticeSurgery(id) | MagicStateDistillation(id) if *id == op.id())
                        {
                            assert_eq!(*steps, 0_u32);
                            continue;
                        }
                        *steps += 1;
                        #[allow(clippy::collapsible_if)]
                        if *steps >= self.conf.magic_state_distillation_cost {
                            let success_rate = self.conf.magic_state_distillation_success_rate;
                            if rng.gen_range(0.0..1.0) < success_rate {
                                *has_magic_state = true;
                            }
                            *steps = 0;
                        }
                    }

                    // Add delay for qubits that are ready for lattice surgery.
                    qubits.for_each(|pos| {
                        let cycle_on_schedule = self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                        let occupancy = &self.schedule[(pos.x, pos.y, cycle_on_schedule)];
                        if matches!(occupancy, LatticeSurgery(id) | DataQubitInOperation(id) if *id == op.id()) {
                            self.delay_at[(pos.x, pos.y)] += 1;
                        }
                    });
                }
                PiOver8RotationState::LatticeSurgery { steps } => {
                    for pos in qubits {
                        let cycle_on_schedule = self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                        let occupancy = &self.schedule[(pos.x, pos.y, cycle_on_schedule)];
                        assert!(
                            matches!(occupancy, LatticeSurgery(id) | DataQubitInOperation(id) if *id == op.id())
                        );
                    }
                    *steps += 1;
                }
                PiOver8RotationState::Correction { steps } => {
                    *steps += 1;
                }
            }
        }
    }

    fn run_internal<R: Rng>(&mut self, rng: &mut R) -> u32 {
        let width = self.schedule.width;
        let height = self.schedule.height;
        loop {
            let runtime_cycle = self.runtime_cycle;
            for (x, y) in range_2d(width, height) {
                // Decrease the delay on idle qubits.
                while runtime_cycle - self.delay_at[(x, y)] < self.end_cycle_at[(x, y)] {
                    let cycle_on_schedule = runtime_cycle - self.delay_at[(x, y)];
                    let occupancy = &self.schedule[(x, y, cycle_on_schedule)];
                    if occupancy.is_vacant_or_idle() && self.delay_at[(x, y)] > 0 {
                        self.delay_at[(x, y)] -= 1;
                    } else {
                        break;
                    }
                }

                let cycle_on_schedule = runtime_cycle - self.delay_at[(x, y)];
                if cycle_on_schedule < self.end_cycle_at[(x, y)] {
                    let occupancy = self.schedule[(x, y, cycle_on_schedule)].clone();
                    self.register_initial_state(occupancy);
                }
            }

            self.perform_pi_over_4_rotation_state_transition();
            self.perform_pi_over_8_rotation_state_transition();
            self.process_pi_over_4_rotation();
            self.process_pi_over_8_rotation(rng);

            if range_2d(width, height).all(|(x, y)| {
                let cycle_on_schedule = runtime_cycle - self.delay_at[(x, y)];
                cycle_on_schedule >= self.end_cycle_at[(x, y)]
            }) {
                break;
            }
            self.runtime_cycle += 1;
        }
        let scheduled_end_cycle = range_2d(width, height)
            .map(|(x, y)| self.end_cycle_at[(x, y)])
            .max()
            .unwrap();
        self.runtime_cycle - scheduled_end_cycle
    }

    pub fn run(&mut self) -> u32 {
        let mut rng = thread_rng();
        self.run_internal(&mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mapping::{DataQubitMapping, Qubit},
        pbc::Pauli,
    };
    use std::ops::Range;

    struct RngForTesting {
        data: Vec<u64>,
        counter: u64,
    }

    impl rand::RngCore for RngForTesting {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }

        fn next_u64(&mut self) -> u64 {
            let ret = self.data[self.counter as usize % self.data.len()];
            self.counter += 1;
            ret
        }

        fn fill_bytes(&mut self, _dest: &mut [u8]) {
            unimplemented!()
        }

        fn try_fill_bytes(&mut self, _dest: &mut [u8]) -> Result<(), rand::Error> {
            unimplemented!()
        }
    }

    impl RngForTesting {
        fn new(data: &[u64]) -> Self {
            RngForTesting {
                data: data.to_vec(),
                counter: 0,
            }
        }

        fn new_with_zero() -> Self {
            RngForTesting {
                data: vec![0],
                counter: 0,
            }
        }

        // The program would crash when `next_64` or `next_u32` is called for the returned generator.
        fn new_unusable() -> Self {
            RngForTesting {
                data: vec![],
                counter: 0,
            }
        }
    }

    fn new_runner(
        operations: Vec<OperationWithAdditionalData>,
        schedule: OccupancyMap,
        conf: &Configuration,
    ) -> Runner {
        let width = schedule.width;
        let height = schedule.height;
        let mut end_cycle_at = Map2D::new_with_value(width, height, 0);
        for (x, y) in range_2d(width, height) {
            let mut cycle = schedule.end_cycle();
            while cycle > 0 {
                if !schedule[(x, y, cycle - 1)].is_vacant_or_idle() {
                    break;
                }
                cycle -= 1;
            }
            end_cycle_at[(x, y)] = cycle;
        }
        Runner {
            operations: operations.iter().map(|op| (op.id(), op.clone())).collect(),
            schedule,
            conf: conf.clone(),
            delay_at: Map2D::new_with_value(width, height, 0),
            end_cycle_at,
            runtime_cycle: 0,
            pi_over_4_rotation_states: HashMap::new(),
            pi_over_8_rotation_states: HashMap::new(),
            removed_operation_ids: HashSet::new(),
        }
    }

    fn set_occupancy(
        map: &mut OccupancyMap,
        x: u32,
        y: u32,
        cycle_range: Range<u32>,
        occupancy: BoardOccupancy,
    ) {
        for cycle in cycle_range {
            map[(x, y, cycle)] = occupancy.clone();
        }
    }

    fn new_occupancy_map(
        width: u32,
        height: u32,
        end_cycle: u32,
        qubit_positions: &[(u32, u32)],
    ) -> OccupancyMap {
        let mut map = OccupancyMap {
            width,
            height,
            map: vec![BoardOccupancy::Vacant; (width * height * end_cycle) as usize],
        };

        for y in 0..height {
            for x in 0..width {
                let occupancy = if qubit_positions.contains(&(x, y)) {
                    BoardOccupancy::IdleDataQubit
                } else {
                    BoardOccupancy::Vacant
                };
                set_occupancy(&mut map, x, y, 0..end_cycle, occupancy);
            }
        }

        map
    }

    fn p(x: u32, y: u32) -> Position {
        Position { x, y }
    }

    #[test]
    fn test_empty() {
        let end_cycle = 30_u32;
        let mut rng = RngForTesting::new_unusable();
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 3);

        let operations = vec![];
        let schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_only_clifford() {
        use BoardOccupancy::*;
        let end_cycle = 9_u32;
        let mut rng = RngForTesting::new_unusable();
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 3);

        let id = OperationId::new(0);
        let operations = vec![OperationWithAdditionalData::PiOver4Rotation {
            id,
            targets: vec![(p(0, 0), Pauli::Z)],
            ancilla_qubits: vec![p(0, 1)],
        }];

        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 0, 0, 0..5, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 0, 1, 0..5, LatticeSurgery(id));
        set_occupancy(&mut schedule, 0, 1, 5..9, YMeasurement(id));

        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_with_pi_over_8_rotations_without_delay() {
        use BoardOccupancy::*;
        let end_cycle = 22_u32;
        let mut rng = RngForTesting::new_with_zero();
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 3);

        let id = OperationId::new(0);
        let operations = vec![OperationWithAdditionalData::PiOver8Rotation {
            id,
            num_distillations: 3,
            num_distillations_on_retry: 3,
            targets: vec![(p(0, 0), Pauli::Z)],
            routing_qubits: vec![],
            distillation_qubits: vec![p(0, 1), p(0, 2), p(1, 2)],
        }];

        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 0, 0, 13..18, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 0, 1, 0..13, MagicStateDistillation(id));
        set_occupancy(&mut schedule, 0, 1, 13..18, LatticeSurgery(id));
        set_occupancy(&mut schedule, 0, 1, 18..22, YMeasurement(id));
        set_occupancy(&mut schedule, 0, 2, 0..13, MagicStateDistillation(id));
        set_occupancy(&mut schedule, 0, 2, 13..18, LatticeSurgery(id));
        set_occupancy(&mut schedule, 0, 2, 18..22, YMeasurement(id));
        set_occupancy(&mut schedule, 1, 2, 0..13, MagicStateDistillation(id));
        set_occupancy(&mut schedule, 1, 2, 13..18, LatticeSurgery(id));
        set_occupancy(&mut schedule, 1, 2, 18..22, YMeasurement(id));

        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_with_pi_over_8_rotations_with_delay_with_one_distillation_qubit() {
        use BoardOccupancy::*;
        let end_cycle = 22_u32;
        let mut rng = RngForTesting::new(&[u64::MAX, u64::MAX, u64::MAX, u64::MAX, 0]);
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 3);

        let id = OperationId::new(0);
        let operations = vec![OperationWithAdditionalData::PiOver8Rotation {
            id,
            num_distillations: 3,
            num_distillations_on_retry: 3,
            targets: vec![(p(0, 0), Pauli::Z)],
            routing_qubits: vec![],
            distillation_qubits: vec![p(0, 1)],
        }];

        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 0, 0, 13..18, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 0, 1, 0..13, MagicStateDistillation(id));
        set_occupancy(&mut schedule, 0, 1, 13..18, LatticeSurgery(id));
        set_occupancy(&mut schedule, 0, 1, 18..22, YMeasurement(id));

        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 52);
        assert_eq!(rng.counter, 5);
    }

    #[test]
    fn test_with_pi_over_8_rotations_with_delay() {
        use BoardOccupancy::*;
        let end_cycle = 22_u32;
        let mut rng = RngForTesting::new(&[u64::MAX, u64::MAX, u64::MAX, u64::MAX, 0]);
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 3);

        let id = OperationId::new(0);
        let operations = vec![OperationWithAdditionalData::PiOver8Rotation {
            id,
            num_distillations: 3,
            num_distillations_on_retry: 3,
            targets: vec![(p(0, 0), Pauli::Z)],
            routing_qubits: vec![],
            distillation_qubits: vec![p(0, 1), p(0, 2), p(1, 2)],
        }];

        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 0, 0, 13..18, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 0, 1, 0..13, MagicStateDistillation(id));
        set_occupancy(&mut schedule, 0, 1, 13..18, LatticeSurgery(id));
        set_occupancy(&mut schedule, 0, 1, 18..22, YMeasurement(id));
        set_occupancy(&mut schedule, 0, 2, 0..13, MagicStateDistillation(id));
        set_occupancy(&mut schedule, 0, 2, 13..18, LatticeSurgery(id));
        set_occupancy(&mut schedule, 0, 2, 18..22, YMeasurement(id));
        set_occupancy(&mut schedule, 1, 2, 0..13, MagicStateDistillation(id));
        set_occupancy(&mut schedule, 1, 2, 13..18, LatticeSurgery(id));
        set_occupancy(&mut schedule, 1, 2, 18..22, YMeasurement(id));

        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 13);
        assert_eq!(rng.counter, 6);
    }

    #[test]
    fn test_magic_state_distillation_retry() {
        use BoardOccupancy::*;
        let end_cycle = 35_u32;
        let mut rng = RngForTesting::new(&[
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            0,
        ]);
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 3);

        let id = OperationId::new(0);
        let operations = vec![OperationWithAdditionalData::PiOver8Rotation {
            id,
            num_distillations: 3,
            num_distillations_on_retry: 2,
            targets: vec![(p(0, 0), Pauli::Z)],
            routing_qubits: vec![],
            distillation_qubits: vec![p(0, 1), p(0, 2)],
        }];

        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 0, 0, 26..31, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 0, 1, 0..26, MagicStateDistillation(id));
        set_occupancy(&mut schedule, 0, 1, 26..31, LatticeSurgery(id));
        set_occupancy(&mut schedule, 0, 1, 31..35, YMeasurement(id));
        set_occupancy(&mut schedule, 0, 2, 0..26, MagicStateDistillation(id));
        set_occupancy(&mut schedule, 0, 2, 26..31, LatticeSurgery(id));
        set_occupancy(&mut schedule, 0, 2, 31..35, YMeasurement(id));

        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 26);
        assert_eq!(rng.counter, 8);
    }

    #[test]
    fn test_delay_propagation() {
        use BoardOccupancy::*;
        let end_cycle = 49_u32;
        let mut rng = RngForTesting::new(&[
            // First three distillations for id 0 starting at cycle 0 => all fail
            u64::MAX,
            u64::MAX,
            u64::MAX,
            // Next three distillations for id0 starting at cycle 13 => one succeeds
            u64::MAX,
            0,
            u64::MAX,
            // First two distillations for id2 (because (2, 1) is delayed) starting at cycle 27 => all fail
            u64::MAX,
            u64::MAX,
            // Next three distillations starting at cycle 40 => fail
            u64::MAX,
            u64::MAX,
            u64::MAX,
            // Next three distillations for id2 starting at cycle 53 => all succeed
            0,
            0,
            0,
        ]);
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 2);

        let id0 = OperationId::new(0);
        let id1 = OperationId::new(1);
        let id2 = OperationId::new(2);
        let operations = vec![
            OperationWithAdditionalData::PiOver8Rotation {
                id: id0,
                num_distillations: 3,
                num_distillations_on_retry: 3,
                targets: vec![(p(0, 0), Pauli::Z)],
                routing_qubits: vec![],
                distillation_qubits: vec![p(0, 1), p(1, 1), p(0, 2)],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: id1,
                targets: vec![(p(2, 2), Pauli::Z)],
                ancilla_qubits: vec![p(1, 1), p(2, 1)],
            },
            OperationWithAdditionalData::PiOver8Rotation {
                id: id2,
                num_distillations: 3,
                num_distillations_on_retry: 2,
                targets: vec![(p(0, 0), Pauli::X)],
                routing_qubits: vec![],
                distillation_qubits: vec![p(1, 0), p(2, 0), p(2, 1)],
            },
        ];

        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 0, 0, 13..18, DataQubitInOperation(id0));
        set_occupancy(&mut schedule, 0, 1, 0..13, MagicStateDistillation(id0));
        set_occupancy(&mut schedule, 0, 1, 13..18, LatticeSurgery(id0));
        set_occupancy(&mut schedule, 0, 1, 18..22, YMeasurement(id0));
        set_occupancy(&mut schedule, 1, 1, 0..13, MagicStateDistillation(id0));
        set_occupancy(&mut schedule, 1, 1, 13..18, LatticeSurgery(id0));
        set_occupancy(&mut schedule, 1, 1, 18..22, YMeasurement(id0));
        set_occupancy(&mut schedule, 0, 2, 0..13, MagicStateDistillation(id0));
        set_occupancy(&mut schedule, 0, 2, 13..18, LatticeSurgery(id0));
        set_occupancy(&mut schedule, 0, 2, 18..22, YMeasurement(id0));

        set_occupancy(&mut schedule, 1, 1, 22..27, LatticeSurgery(id1));
        set_occupancy(&mut schedule, 1, 1, 27..31, YMeasurement(id1));
        set_occupancy(&mut schedule, 2, 1, 22..27, LatticeSurgery(id1));
        set_occupancy(&mut schedule, 2, 2, 22..27, DataQubitInOperation(id1));

        set_occupancy(&mut schedule, 0, 0, 40..45, DataQubitInOperation(id2));
        set_occupancy(&mut schedule, 1, 0, 27..40, MagicStateDistillation(id2));
        set_occupancy(&mut schedule, 1, 0, 40..45, LatticeSurgery(id2));
        set_occupancy(&mut schedule, 1, 0, 45..49, YMeasurement(id2));
        set_occupancy(&mut schedule, 2, 0, 27..40, MagicStateDistillation(id2));
        set_occupancy(&mut schedule, 2, 0, 40..45, LatticeSurgery(id2));
        set_occupancy(&mut schedule, 2, 0, 45..49, YMeasurement(id2));
        set_occupancy(&mut schedule, 2, 1, 27..40, MagicStateDistillation(id2));
        set_occupancy(&mut schedule, 2, 1, 40..45, LatticeSurgery(id2));
        set_occupancy(&mut schedule, 2, 1, 45..49, YMeasurement(id2));

        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 26);
        assert_eq!(rng.counter, 14);
    }

    #[test]
    fn test_delay_propagation_through_data_qubit() {
        use BoardOccupancy::*;
        let end_cycle = 32_u32;
        let mut rng = RngForTesting::new(&[u64::MAX, 0]);
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 1, 1);
        mapping.map(q1, 2, 2);

        let id0 = OperationId::new(0);
        let id1 = OperationId::new(1);
        let id2 = OperationId::new(2);
        let operations = vec![
            OperationWithAdditionalData::PiOver8Rotation {
                id: id0,
                num_distillations: 1,
                num_distillations_on_retry: 1,
                targets: vec![(p(1, 1), Pauli::Z)],
                routing_qubits: vec![],
                distillation_qubits: vec![p(1, 2)],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: id1,
                targets: vec![(p(1, 1), Pauli::X)],
                ancilla_qubits: vec![p(0, 1)],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: id2,
                targets: vec![(p(1, 1), Pauli::X)],
                ancilla_qubits: vec![p(2, 1)],
            },
        ];

        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 1, 1, 13..18, DataQubitInOperation(id0));
        set_occupancy(&mut schedule, 1, 2, 0..13, MagicStateDistillation(id0));
        set_occupancy(&mut schedule, 1, 2, 13..18, LatticeSurgery(id0));
        set_occupancy(&mut schedule, 1, 2, 18..22, YMeasurement(id0));

        set_occupancy(&mut schedule, 1, 1, 18..23, DataQubitInOperation(id1));
        set_occupancy(&mut schedule, 0, 1, 18..23, LatticeSurgery(id1));
        set_occupancy(&mut schedule, 0, 1, 23..27, YMeasurement(id1));

        set_occupancy(&mut schedule, 1, 1, 23..28, DataQubitInOperation(id2));
        set_occupancy(&mut schedule, 2, 1, 23..28, LatticeSurgery(id2));
        set_occupancy(&mut schedule, 2, 1, 28..32, YMeasurement(id2));

        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 13);
        assert_eq!(rng.counter, 2);
    }

    #[test]
    fn test_delay_reduction_with_vacant() {
        use BoardOccupancy::*;
        let end_cycle = 33_u32;
        let mut rng = RngForTesting::new(&[u64::MAX, u64::MAX, u64::MAX, u64::MAX, 0]);
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 2);

        let id0 = OperationId::new(0);
        let id1 = OperationId::new(1);
        let operations = vec![
            OperationWithAdditionalData::PiOver8Rotation {
                id: id0,
                num_distillations: 3,
                num_distillations_on_retry: 3,
                targets: vec![(p(0, 0), Pauli::Z)],
                routing_qubits: vec![],
                distillation_qubits: vec![p(0, 1), p(1, 1), p(0, 2)],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: id1,
                targets: vec![(p(2, 2), Pauli::Z)],
                ancilla_qubits: vec![p(1, 1), p(2, 1)],
            },
        ];

        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 0, 0, 13..18, DataQubitInOperation(id0));
        set_occupancy(&mut schedule, 0, 1, 0..13, MagicStateDistillation(id0));
        set_occupancy(&mut schedule, 0, 1, 13..18, LatticeSurgery(id0));
        set_occupancy(&mut schedule, 0, 1, 18..22, YMeasurement(id0));
        set_occupancy(&mut schedule, 1, 1, 0..13, MagicStateDistillation(id0));
        set_occupancy(&mut schedule, 1, 1, 13..18, LatticeSurgery(id0));
        set_occupancy(&mut schedule, 1, 1, 18..22, YMeasurement(id0));
        set_occupancy(&mut schedule, 0, 2, 0..13, MagicStateDistillation(id0));
        set_occupancy(&mut schedule, 0, 2, 13..18, LatticeSurgery(id0));
        set_occupancy(&mut schedule, 0, 2, 18..22, YMeasurement(id0));

        set_occupancy(&mut schedule, 1, 1, 24..29, LatticeSurgery(id1));
        set_occupancy(&mut schedule, 1, 1, 29..33, YMeasurement(id1));
        set_occupancy(&mut schedule, 2, 1, 24..29, LatticeSurgery(id1));
        set_occupancy(&mut schedule, 2, 2, 24..29, DataQubitInOperation(id1));

        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 11);
        assert_eq!(rng.counter, 6);
    }

    #[test]
    fn test_delay_reduction_with_idle() {
        use BoardOccupancy::*;
        let end_cycle = 34_u32;
        let mut rng = RngForTesting::new(&[u64::MAX, 0]);
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 1, 1);
        mapping.map(q1, 2, 2);

        let id0 = OperationId::new(0);
        let id1 = OperationId::new(1);
        let id2 = OperationId::new(2);
        let operations = vec![
            OperationWithAdditionalData::PiOver8Rotation {
                id: id0,
                num_distillations: 1,
                num_distillations_on_retry: 1,
                targets: vec![(Position { x: 1, y: 1 }, Pauli::Z)],
                routing_qubits: vec![],
                distillation_qubits: vec![p(1, 2)],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: id1,
                targets: vec![(Position { x: 1, y: 1 }, Pauli::X)],
                ancilla_qubits: vec![p(0, 1)],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: id2,
                targets: vec![(Position { x: 1, y: 1 }, Pauli::X)],
                ancilla_qubits: vec![p(2, 1)],
            },
        ];

        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 1, 1, 13..18, DataQubitInOperation(id0));
        set_occupancy(&mut schedule, 1, 2, 0..13, MagicStateDistillation(id0));
        set_occupancy(&mut schedule, 1, 2, 13..18, LatticeSurgery(id0));
        set_occupancy(&mut schedule, 1, 2, 18..22, YMeasurement(id0));

        set_occupancy(&mut schedule, 1, 1, 20..25, DataQubitInOperation(id1));
        set_occupancy(&mut schedule, 0, 1, 20..25, LatticeSurgery(id1));
        set_occupancy(&mut schedule, 0, 1, 25..29, YMeasurement(id1));

        set_occupancy(&mut schedule, 1, 1, 25..30, DataQubitInOperation(id2));
        set_occupancy(&mut schedule, 2, 1, 25..30, LatticeSurgery(id2));
        set_occupancy(&mut schedule, 2, 1, 30..34, YMeasurement(id2));

        let mut runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 11);
        assert_eq!(rng.counter, 2);
    }
}
