use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
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

#[derive(Clone, Debug)]
pub struct OccupancyMap {
    width: u32,
    height: u32,
    map: Vec<BoardOccupancy>,
}

impl Index<(u32, u32, u32)> for OccupancyMap {
    type Output = BoardOccupancy;

    fn index(&self, (x, y, cycle): (u32, u32, u32)) -> &Self::Output {
        assert!(x < self.width);
        assert!(y < self.height);
        let index = (cycle * self.width * self.height + y * self.width + x) as usize;
        &self.map[index]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PatchDirection {
    HorizontalZBoundary,
    VerticalZBoundary,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
enum PiOver4RotationState {
    Initial,
    LatticeSurgery { steps: u32 },
    Correction { steps: u32 },
}

// Represents the distillation state in SingleQubitPiOver8RotationBlock.
#[derive(Clone, Debug, Eq, PartialEq)]
struct DistillationState {
    position: Position,
    steps: u32,
    direction: PatchDirection,
    is_involved_with_lattice_surgery: bool,
}

impl DistillationState {
    fn new(position: Position, direction: PatchDirection) -> Self {
        DistillationState {
            position,
            steps: 0,
            direction,
            is_involved_with_lattice_surgery: false,
        }
    }

    fn renew(&mut self) {
        self.steps = 0;
        self.is_involved_with_lattice_surgery = false;
    }

    fn mark_as_involved_with_lattice_surgery(&mut self) {
        self.is_involved_with_lattice_surgery = true;
        self.steps = 0;
    }

    fn is_ready(&self, conf: &Configuration) -> bool {
        self.steps == conf.magic_state_distillation_cost
    }

    fn run_distillation<R: Rng>(&mut self, rng: &mut R, conf: &Configuration) {
        if self.is_ready(conf) {
            return;
        }
        if self.is_involved_with_lattice_surgery {
            return;
        }
        self.steps += 1;
        if self.steps == conf.magic_state_distillation_cost {
            let success_rate = conf.magic_state_distillation_success_rate;
            if rng.gen_range(0.0..1.0) > success_rate {
                // Distillation failed.
                self.steps = 0;
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum SingleQubitPiOver8RotationBlockState {
    PiOver8Rotation {
        // Represents the pi/8 rotation being performed.
        index: u32,
        // None if lattice surgery is not being performed.
        lattice_surgery_steps: Option<u32>,
        distillation_steps: Vec<DistillationState>,
    },
    Correction {
        steps: u32,
    },
}

#[derive(Clone, Debug)]
pub struct Runner {
    operations: HashMap<OperationId, OperationWithAdditionalData>,
    schedule: OccupancyMap,
    conf: Configuration,

    delay_at: Map2D<u32>,
    end_cycle_at: Map2D<u32>,
    runtime_cycle: u32,
    pi_over_4_rotation_states: HashMap<OperationId, PiOver4RotationState>,
    pi_over_8_rotation_states: HashMap<OperationId, PiOver8RotationState>,
    pi_over_8_rotation_block_states: HashMap<OperationId, SingleQubitPiOver8RotationBlockState>,
    removed_operation_ids: HashSet<OperationId>,

    // This is for checking / debugging.
    end_cycle_for_pi_over_8_rotation_block: HashMap<OperationId, u32>,
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

        let mut runner = Runner {
            operations,
            schedule,
            conf: board.configuration().clone(),
            delay_at: Map2D::new_with_value(board.width(), board.height(), 0),
            end_cycle_at,
            runtime_cycle: 0,
            pi_over_4_rotation_states: HashMap::new(),
            pi_over_8_rotation_states: HashMap::new(),
            pi_over_8_rotation_block_states: HashMap::new(),
            removed_operation_ids: HashSet::new(),
            end_cycle_for_pi_over_8_rotation_block: HashMap::new(),
        };

        // Set this to true when you are debugging.
        let with_checking_end_cycle_for_pi_over_8_rotation_blocks = false;
        if with_checking_end_cycle_for_pi_over_8_rotation_blocks {
            runner.end_cycle_for_pi_over_8_rotation_block =
                runner.construct_end_cycle_for_pi_over_8_rotation_block();
        }

        runner
    }

    fn construct_end_cycle_for_pi_over_8_rotation_block(&self) -> HashMap<OperationId, u32> {
        let width = self.conf.width;
        let height = self.conf.height;
        let mut end_cycle_for_pi_over_8_rotation_block = HashMap::<OperationId, u32>::new();
        for (_id, op) in &self.operations {
            use OperationWithAdditionalData::SingleQubitPiOver8RotationBlock;
            let (target, routing_qubits, distillation_qubits, correction_qubits) =
                if let SingleQubitPiOver8RotationBlock {
                    target,
                    routing_qubits,
                    distillation_qubits,
                    correction_qubits,
                    ..
                } = op
                {
                    (target, routing_qubits, distillation_qubits, correction_qubits)
                } else {
                    continue;
                };
            assert_eq!(correction_qubits.len(), 3);
            assert_eq!(routing_qubits.len(), 3);
            assert!(distillation_qubits.contains(&correction_qubits[0]));
            assert!(routing_qubits.contains(&correction_qubits[1]));
            assert!(routing_qubits.contains(&correction_qubits[2]));

            let cq0 = correction_qubits[0];
            let cq1 = correction_qubits[1];
            let cq2 = correction_qubits[2];
            let range = 0..self.end_cycle_at[(cq1.x, cq1.y)];
            let end_cycle = range
                .rev()
                .find(|&cycle| self.schedule[(cq1.x, cq1.y, cycle)].operation_id() == Some(op.id()))
                .unwrap()
                + 1;
            let distance = self.conf.code_distance;
            let y_measurement_cost = y_measurement_cost(distance);
            assert!(end_cycle > 2 * distance + y_measurement_cost);
            for cycle in end_cycle - y_measurement_cost..end_cycle {
                for (x, y) in range_2d(width, height) {
                    let occupancy = &self.schedule[(x, y, cycle)];
                    if (x, y) == (cq1.x, cq1.y) || (x, y) == (cq2.x, cq2.y) {
                        // The last Y measurement is performed at one of these correction qubits.
                        assert_eq!(*occupancy, BoardOccupancy::PiOver8RotationBlock(op.id()));
                    } else {
                        assert_ne!(*occupancy, BoardOccupancy::PiOver8RotationBlock(op.id()));
                        assert_ne!(*occupancy, BoardOccupancy::DataQubitInOperation(op.id()));
                    }
                }
            }
            for cycle in end_cycle - y_measurement_cost - distance..end_cycle - y_measurement_cost {
                for (x, y) in range_2d(width, height) {
                    let occupancy = &self.schedule[(x, y, cycle)];
                    if (x, y) == (cq0.x, cq0.y) {
                        // The second-last Y measurement is performed at this correction qubit.
                        if cycle - (end_cycle - y_measurement_cost - distance) < y_measurement_cost
                        {
                            assert_eq!(*occupancy, BoardOccupancy::PiOver8RotationBlock(op.id()));
                        } else {
                            assert_ne!(*occupancy, BoardOccupancy::PiOver8RotationBlock(op.id()));
                        }
                    } else if (x, y) == (target.x, target.y) {
                        assert_eq!(*occupancy, BoardOccupancy::DataQubitInOperation(op.id()));
                    } else if routing_qubits.contains(&Position::new(x, y)) {
                        assert_eq!(*occupancy, BoardOccupancy::PiOver8RotationBlock(op.id()));
                    } else {
                        assert_ne!(*occupancy, BoardOccupancy::PiOver8RotationBlock(op.id()));
                        assert_ne!(*occupancy, BoardOccupancy::DataQubitInOperation(op.id()));
                    }
                }
            }
            for cycle in end_cycle - y_measurement_cost - 2 * distance
                ..end_cycle - y_measurement_cost - distance
            {
                for (x, y) in range_2d(width, height) {
                    let occupancy = &self.schedule[(x, y, cycle)];
                    if (x, y) == (target.x, target.y) {
                        assert_eq!(*occupancy, BoardOccupancy::DataQubitInOperation(op.id()));
                    } else if routing_qubits.contains(&Position::new(x, y))
                        || distillation_qubits.contains(&Position::new(x, y))
                    {
                        assert_eq!(*occupancy, BoardOccupancy::PiOver8RotationBlock(op.id()));
                    } else {
                        assert_ne!(*occupancy, BoardOccupancy::PiOver8RotationBlock(op.id()));
                        assert_ne!(*occupancy, BoardOccupancy::DataQubitInOperation(op.id()));
                    }
                }
            }
            end_cycle_for_pi_over_8_rotation_block.insert(op.id(), end_cycle);
        }
        end_cycle_for_pi_over_8_rotation_block
    }

    fn new_distillation_states(
        target: &Position,
        routing_qubits: &[Position],
        distillation_qubits: &[Position],
    ) -> Vec<DistillationState> {
        use PatchDirection::*;
        let mut states = vec![];

        // We use Vector here because we assume the number of qubits is not so large.
        let mut visited_qubits = routing_qubits
            .iter()
            .chain(std::iter::once(target))
            .cloned()
            .collect::<Vec<_>>();

        let mut q = VecDeque::new();
        for pos in distillation_qubits {
            if pos.x > 0 && routing_qubits.contains(&Position::new(pos.x - 1, pos.y)) {
                q.push_back((*pos, VerticalZBoundary));
            }
            if pos.y > 0 && routing_qubits.contains(&Position::new(pos.x, pos.y - 1)) {
                q.push_back((*pos, HorizontalZBoundary));
            }
            if routing_qubits.contains(&Position::new(pos.x + 1, pos.y)) {
                q.push_back((*pos, VerticalZBoundary));
            }
            if routing_qubits.contains(&Position::new(pos.x, pos.y + 1)) {
                q.push_back((*pos, HorizontalZBoundary));
            }
        }

        while let Some((pos, direction)) = q.pop_front() {
            if visited_qubits.contains(&pos) {
                continue;
            }
            visited_qubits.push(pos);
            states.push(DistillationState::new(pos, direction));

            if pos.y > 0 && distillation_qubits.contains(&Position::new(pos.x, pos.y - 1)) {
                q.push_back((Position::new(pos.x, pos.y - 1), HorizontalZBoundary));
            }
            if pos.x > 0 && distillation_qubits.contains(&Position::new(pos.x - 1, pos.y)) {
                q.push_back((Position::new(pos.x - 1, pos.y), VerticalZBoundary));
            }
            if distillation_qubits.contains(&Position::new(pos.x + 1, pos.y)) {
                q.push_back((Position::new(pos.x + 1, pos.y), VerticalZBoundary));
            }
            if distillation_qubits.contains(&Position::new(pos.x, pos.y + 1)) {
                q.push_back((Position::new(pos.x, pos.y + 1), HorizontalZBoundary));
            }
        }

        let mut positions_in_states = states.iter().map(|s| s.position).collect::<Vec<_>>();
        positions_in_states.sort();
        let mut sorted_distillation_qubits = distillation_qubits.to_vec();
        sorted_distillation_qubits.sort();
        assert_eq!(sorted_distillation_qubits, positions_in_states);

        states
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
                SingleQubitPiOver8RotationBlock {
                    target,
                    routing_qubits,
                    distillation_qubits,
                    ..
                } => {
                    self.pi_over_8_rotation_block_states
                        .entry(id)
                        .or_insert_with(|| {
                            let distillation_states = Self::new_distillation_states(
                                target,
                                routing_qubits,
                                distillation_qubits,
                            );
                            SingleQubitPiOver8RotationBlockState::PiOver8Rotation {
                                index: 0,
                                lattice_surgery_steps: None,
                                distillation_steps: distillation_states,
                            }
                        });
                }
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

    fn perform_pi_over_8_rotation_block_state_transition(&mut self) {
        use OperationWithAdditionalData::*;
        let mut to_be_removed = vec![];
        for (id, state) in &mut self.pi_over_8_rotation_block_states {
            let op = &self.operations[id];
            let pi_over_8_axes = match op {
                SingleQubitPiOver8RotationBlock { pi_over_8_axes, .. } => pi_over_8_axes,
                _ => unreachable!(),
            };
            loop {
                match state {
                    SingleQubitPiOver8RotationBlockState::PiOver8Rotation {
                        index,
                        lattice_surgery_steps,
                        distillation_steps,
                    } => {
                        if let Some(steps) = lattice_surgery_steps {
                            if *steps == self.conf.code_distance {
                                for site in distillation_steps.iter_mut() {
                                    if site.is_involved_with_lattice_surgery {
                                        site.renew();
                                    }
                                }
                                *lattice_surgery_steps = None;
                                *index += 1;
                            }
                        }
                        if *index as usize == pi_over_8_axes.len() {
                            *state = SingleQubitPiOver8RotationBlockState::Correction { steps: 0 };
                            continue;
                        }
                    }
                    SingleQubitPiOver8RotationBlockState::Correction { steps } => {
                        let distance = self.conf.code_distance;
                        if *steps == 2 * distance + y_measurement_cost(distance) {
                            to_be_removed.push(*id);
                        }
                    }
                }
                break;
            }
        }
        self.removed_operation_ids.extend(to_be_removed.iter());
        for id in to_be_removed {
            self.pi_over_8_rotation_block_states.remove(&id);
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
                        if *steps >= self.conf.magic_state_distillation_cost {
                            let success_rate = self.conf.magic_state_distillation_success_rate;
                            if rng.gen_range(0.0..1.0) <= success_rate {
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
                            matches!(occupancy, LatticeSurgery(id) | DataQubitInOperation(id) | MagicStateDistillation(id) if *id == op.id())
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

    // Calculates the minimal cost to transfer a magic state to the routing area, and returns
    // the distillation state and the path, if there is one.
    // The path does not include both endpoints (the routing area and the magic state).
    // Every position in the returned path must be contained in `available_positions`.
    fn get_ready_magic_state<'a>(
        routing_qubits: &[Position],
        distillation_steps: &'a mut [DistillationState],
        available_positions: &[Position],
        conf: &Configuration,
    ) -> Option<(&'a mut DistillationState, Vec<Position>)> {
        use PatchDirection::*;
        #[derive(Clone, Debug, Eq, PartialEq)]
        struct State {
            cost: u32,
            distance: u32,
            position: Position,
            prev: Option<usize>,
        }
        impl State {
            fn new(cost: u32, distance: u32, position: Position, prev: Option<usize>) -> Self {
                State {
                    cost,
                    distance,
                    position,
                    prev,
                }
            }
        }
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // We use `reverse` here because BinaryHeap is a max heap.
                self.cost.cmp(&other.cost).reverse()
            }
        }

        let mut q = BinaryHeap::new();

        for pos in routing_qubits {
            q.push(State::new(0, 0, *pos, None));
        }

        let mut cost_table: Vec<State> = vec![];
        while let Some(state) = q.pop() {
            if cost_table.iter().any(|s| s.position == state.position) {
                continue;
            }
            let prev = cost_table.len();
            cost_table.push(state.clone());

            let mut push = |pos| {
                if !available_positions.contains(&pos) {
                    return;
                }
                let index = match distillation_steps.iter().position(|s| s.position == pos) {
                    Some(index) => index,
                    None => return,
                };
                let distillation_state = &distillation_steps[index];
                let failure_rate = 1.0 - conf.magic_state_distillation_success_rate;
                let steps = distillation_state.steps;
                let cost_to_discard_distillation = if steps == conf.magic_state_distillation_cost {
                    (steps as f64 / (failure_rate * failure_rate)).ceil() as u32
                } else {
                    steps
                };
                let cost = state.cost + cost_to_discard_distillation;
                let distance = state.distance + 1;

                q.push(State::new(cost, distance, pos, Some(prev)));
            };

            let pos = state.position;
            if pos.x > 0 {
                push(Position::new(pos.x - 1, pos.y));
            }
            if pos.y > 0 {
                push(Position::new(pos.x, pos.y - 1));
            }
            push(Position::new(pos.x + 1, pos.y));
            push(Position::new(pos.x, pos.y + 1));
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct Candidate {
            cost: u32,
            distance: u32,
            index: usize,
            cost_table_index: usize,
        }
        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.cost
                    .cmp(&other.cost)
                    .then(self.distance.cmp(&other.distance).reverse())
            }
        }
        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut candidate: Option<Candidate> = None;
        for (index, state) in distillation_steps.iter().enumerate() {
            if !state.is_ready(conf) {
                continue;
            }
            let mut update_candidate = |pos| {
                let cost_table_index = match cost_table.iter().position(|s| s.position == pos) {
                    Some(index) => index,
                    None => return,
                };

                let cost = cost_table[cost_table_index].cost;
                let distance = cost_table[cost_table_index].distance;
                let new_candidate = Candidate {
                    cost,
                    distance,
                    index,
                    cost_table_index,
                };
                candidate = match &candidate {
                    Some(candidate) => Some(std::cmp::min(candidate.clone(), new_candidate)),
                    None => Some(new_candidate),
                };
            };

            let pos = state.position;
            match state.direction {
                HorizontalZBoundary => {
                    if pos.y > 0 {
                        update_candidate(Position::new(pos.x, pos.y - 1));
                    }
                    update_candidate(Position::new(pos.x, pos.y + 1));
                }
                VerticalZBoundary => {
                    if pos.x > 0 {
                        update_candidate(Position::new(pos.x - 1, pos.y));
                    }
                    update_candidate(Position::new(pos.x + 1, pos.y));
                }
            }
        }

        match candidate {
            None => None,
            Some(candidate) => {
                let mut path = vec![];
                let mut state: &State = &cost_table[candidate.cost_table_index];
                loop {
                    path.push(state.position);
                    state = if let Some(prev) = state.prev {
                        &cost_table[prev]
                    } else {
                        break;
                    };
                }
                assert!(!path.is_empty());
                path.remove(path.len() - 1);
                path.reverse();

                Some((&mut distillation_steps[candidate.index], path))
            }
        }
    }

    fn process_pi_over_8_rotation_block<R: Rng>(&mut self, rng: &mut R) {
        use BoardOccupancy::*;
        for (id, state) in &mut self.pi_over_8_rotation_block_states {
            let op = &self.operations[id];
            let (target, routing_qubits, distillation_qubits, correction_qubits, pi_over_8_axes) =
                match op {
                    OperationWithAdditionalData::SingleQubitPiOver8RotationBlock {
                        target,
                        routing_qubits,
                        distillation_qubits,
                        correction_qubits,
                        pi_over_8_axes,
                        ..
                    } => (
                        target,
                        routing_qubits,
                        distillation_qubits,
                        correction_qubits,
                        pi_over_8_axes,
                    ),
                    _ => unreachable!(),
                };
            let is_associated_with_this_op = |occupancy: BoardOccupancy| {
                matches!(occupancy, PiOver8RotationBlock(id) if id == op.id())
                    || matches!(occupancy, DataQubitInOperation(id) if id == op.id())
            };

            let next_active_qubits: Vec<Position>;
            match state {
                SingleQubitPiOver8RotationBlockState::PiOver8Rotation {
                    index,
                    lattice_surgery_steps,
                    distillation_steps,
                } => {
                    let is_available = |pos: &Position| {
                        let cycle_on_schedule = self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                        if cycle_on_schedule >= self.end_cycle_at[(pos.x, pos.y)] {
                            return false;
                        }
                        let occupancy = &self.schedule[(pos.x, pos.y, cycle_on_schedule)];
                        is_associated_with_this_op(occupancy.clone())
                    };

                    if let Some(steps) = lattice_surgery_steps {
                        *steps += 1;
                    } else if is_available(target) && routing_qubits.iter().all(is_available) {
                        let available_positions = routing_qubits
                            .iter()
                            .filter(|p| is_available(p))
                            .chain(distillation_qubits.iter().filter(|p| is_available(p)))
                            .cloned()
                            .collect::<Vec<_>>();

                        // Find a path to connect the routing_area with a magic state, if there is one.
                        if let Some((distillation_site, path)) = Self::get_ready_magic_state(
                            routing_qubits,
                            distillation_steps,
                            &available_positions,
                            &self.conf,
                        ) {
                            // Consume the magic state.
                            distillation_site.mark_as_involved_with_lattice_surgery();
                            for site in distillation_steps.iter_mut() {
                                if path.contains(&site.position) {
                                    site.mark_as_involved_with_lattice_surgery();
                                }
                            }
                            *lattice_surgery_steps = Some(1);
                        }
                    }
                    for site in distillation_steps.iter_mut() {
                        let pos = site.position;
                        let cycle_on_schedule = self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                        if cycle_on_schedule < self.end_cycle_at[(pos.x, pos.y)] {
                            let occupancy = &self.schedule[(pos.x, pos.y, cycle_on_schedule)];
                            if matches!(occupancy, PiOver8RotationBlock(id) if *id == op.id()) {
                                site.run_distillation(rng, &self.conf);
                            }
                        }
                    }

                    if *index == pi_over_8_axes.len() as u32 - 1
                        && lattice_surgery_steps == &Some(self.conf.code_distance)
                    {
                        next_active_qubits = std::iter::once(target)
                            .chain(routing_qubits)
                            .chain(std::iter::once(&correction_qubits[0]))
                            .cloned()
                            .collect::<Vec<_>>();
                    } else {
                        next_active_qubits = std::iter::once(target)
                            .chain(routing_qubits.iter())
                            .chain(distillation_qubits.iter())
                            .cloned()
                            .collect::<Vec<_>>();
                    }
                }
                SingleQubitPiOver8RotationBlockState::Correction { steps } => {
                    let distance = self.conf.code_distance;
                    let y_measurement_cost = y_measurement_cost(distance);
                    assert!(y_measurement_cost <= distance);
                    assert_eq!(correction_qubits.len(), 3);
                    *steps += 1;

                    next_active_qubits = if *steps < distance + y_measurement_cost {
                        std::iter::once(target)
                            .chain(routing_qubits)
                            .chain(std::iter::once(&correction_qubits[0]))
                            .cloned()
                            .collect::<Vec<_>>()
                    } else if *steps < 2 * distance {
                        std::iter::once(target)
                            .chain(routing_qubits)
                            .cloned()
                            .collect::<Vec<_>>()
                    } else if *steps < 2 * distance + y_measurement_cost {
                        correction_qubits
                            .iter()
                            .skip(1)
                            .cloned()
                            .collect::<Vec<_>>()
                    } else {
                        vec![]
                    };
                }
            }

            let qubits = std::iter::once(target)
                .chain(routing_qubits)
                .chain(distillation_qubits)
                .cloned()
                .collect::<Vec<_>>();

            for pos in qubits {
                let cycle_on_schedule = self.runtime_cycle - self.delay_at[(pos.x, pos.y)];
                let end_cycle = self.end_cycle_at[(pos.x, pos.y)];
                if cycle_on_schedule >= end_cycle {
                    continue;
                }
                if !is_associated_with_this_op(
                    self.schedule[(pos.x, pos.y, cycle_on_schedule)].clone(),
                ) {
                    // In this case, this operation does not 'own' this position.
                    continue;
                }

                if next_active_qubits.contains(&pos) {
                    let current = self.schedule[(pos.x, pos.y, cycle_on_schedule)].clone();
                    let next = if cycle_on_schedule + 1 < end_cycle {
                        Some(self.schedule[(pos.x, pos.y, cycle_on_schedule + 1)].clone())
                    } else {
                        None
                    };
                    // If the current occupancy is suitable and the next occupancy is not suitable, then
                    // we should add a delay for the next (runtime) cycle.
                    let should_add_delay = is_associated_with_this_op(current)
                        && !next.map_or(false, is_associated_with_this_op);

                    if should_add_delay {
                        self.delay_at[(pos.x, pos.y)] += 1;
                    }
                } else {
                    loop {
                        let next_cycle_on_schedule =
                            self.runtime_cycle + 1 - self.delay_at[(pos.x, pos.y)];
                        if next_cycle_on_schedule >= self.end_cycle_at[(pos.x, pos.y)] {
                            break;
                        }
                        let occupancy =
                            self.schedule[(pos.x, pos.y, next_cycle_on_schedule)].clone();
                        if !is_associated_with_this_op(occupancy) {
                            break;
                        }
                        if self.delay_at[(pos.x, pos.y)] == 0 {
                            break;
                        }
                        self.delay_at[(pos.x, pos.y)] -= 1;
                    }
                }
            }
        }
    }

    // `suppress_delay_reduction` can be true only for testing.
    fn run_internal_step<R: Rng>(&mut self, rng: &mut R, suppress_delay_reduction: bool) {
        let width = self.schedule.width;
        let height = self.schedule.height;
        let runtime_cycle = self.runtime_cycle;
        for (x, y) in range_2d(width, height) {
            if !suppress_delay_reduction {
                // Decrease the delay on idle qubits.
                let delay = &mut self.delay_at[(x, y)];
                while *delay > 0 && runtime_cycle - *delay < self.end_cycle_at[(x, y)] {
                    let cycle_on_schedule = runtime_cycle - *delay;
                    let occupancy = &self.schedule[(x, y, cycle_on_schedule)];
                    let is_idle = occupancy.is_vacant_or_idle()
                        || occupancy
                            .operation_id()
                            .map_or(false, |id| self.removed_operation_ids.contains(&id));
                    if is_idle {
                        *delay -= 1;
                    } else {
                        break;
                    }
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
        self.perform_pi_over_8_rotation_block_state_transition();
        self.process_pi_over_4_rotation();
        self.process_pi_over_8_rotation(rng);
        self.process_pi_over_8_rotation_block(rng);
    }

    fn run_internal<R: Rng>(&mut self, rng: &mut R) -> u32 {
        let width = self.schedule.width;
        let height = self.schedule.height;
        loop {
            self.run_internal_step(rng, false);

            if range_2d(width, height).all(|(x, y)| {
                let cycle_on_schedule = self.runtime_cycle - self.delay_at[(x, y)];
                cycle_on_schedule >= self.end_cycle_at[(x, y)]
            }) {
                break;
            }

            self.check_pi_over_8_rotation_block_ops();
            self.runtime_cycle += 1;

            assert!(self.runtime_cycle < 2_000_000_000);
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

    fn check_pi_over_8_rotation_block_ops(&self) {
        if self.end_cycle_for_pi_over_8_rotation_block.is_empty() {
            return;
        }

        for (id, end_cycle) in &self.end_cycle_for_pi_over_8_rotation_block {
            let state = if let Some(state) = self.pi_over_8_rotation_block_states.get(id) {
                state
            } else {
                continue;
            };
            let (target, routing_qubits, distillation_qubits, pi_over_8_axes) =
                match &self.operations[id] {
                    OperationWithAdditionalData::SingleQubitPiOver8RotationBlock {
                        target,
                        routing_qubits,
                        distillation_qubits,
                        pi_over_8_axes,
                        ..
                    } => (target, routing_qubits, distillation_qubits, pi_over_8_axes),
                    _ => unreachable!(),
                };
            match state {
                SingleQubitPiOver8RotationBlockState::PiOver8Rotation {
                    index,
                    lattice_surgery_steps,
                    ..
                } => {
                    if (*index as usize) == pi_over_8_axes.len() - 1
                        && lattice_surgery_steps == &Some(self.conf.code_distance)
                    {
                        continue;
                    }
                    assert!(!self.removed_operation_ids.contains(id));
                    for (x, y) in range_2d(self.conf.width, self.conf.height) {
                        let cycle_on_schedule = self.runtime_cycle - self.delay_at[(x, y)];
                        // let occupancy = &self.schedule[(x, y, cycle_on_schedule)];
                        let pos = Position::new(x, y);
                        let is_in_block = pos == *target
                            || routing_qubits.contains(&pos)
                            || distillation_qubits.contains(&pos);
                        if is_in_block {
                            if cycle_on_schedule >= *end_cycle {
                                println!(
                                    "id = {:?}, cycle_on_schedule = {}, end_cycle = {}",
                                    id, cycle_on_schedule, end_cycle
                                );
                                println!("delay_at[x, y] = {}", self.delay_at[(x, y)],);
                                println!("x = {}, y = {}, target = {:?}, routing_qubits = {:?}, distillation_qubits = {:?}", x, y, target, routing_qubits, distillation_qubits);
                                println!("index = {}, pi_over_8_axes.len = {}, lattice_surgery_steps = {:?}", index, pi_over_8_axes.len(), lattice_surgery_steps);
                            }
                            assert!(cycle_on_schedule < *end_cycle);
                        }
                    }
                }
                SingleQubitPiOver8RotationBlockState::Correction { .. } => {
                    continue;
                }
            }
        }
    }

    pub fn runtime_cycle(&self) -> u32 {
        self.runtime_cycle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutils::RngForTesting;
    use crate::{
        mapping::{DataQubitMapping, Qubit},
        pbc::Pauli,
    };
    use std::ops::Range;

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
            pi_over_8_rotation_block_states: HashMap::new(),
            removed_operation_ids: HashSet::new(),
            end_cycle_for_pi_over_8_rotation_block: HashMap::new(),
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

    fn default_conf() -> Configuration {
        Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            num_distillations_for_pi_over_8_rotation: 5,
            magic_state_distillation_success_rate: 0.5,
            num_distillations_for_pi_over_8_rotation_block: 3,
            single_qubit_pi_over_8_rotation_block_depth_ratio: 1.2,
            single_qubit_arbitrary_angle_rotation_precision: 1e-10,
            preferable_distillation_area_size: 5,
        }
    }

    #[test]
    fn test_empty() {
        let end_cycle = 30_u32;
        let mut rng = RngForTesting::new_unusable();
        let conf = default_conf();
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
            ..default_conf()
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
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
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
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
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
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
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
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
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
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
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
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
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
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
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
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
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

    #[test]
    fn test_get_ready_magic_state_none() {
        use PatchDirection::*;
        let conf = Configuration {
            width: 4,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };
        let routing_qubits = vec![
            Position::new(1, 0),
            Position::new(2, 0),
            Position::new(2, 1),
        ];
        let mut distillation_steps = vec![
            DistillationState {
                position: Position::new(3, 0),
                steps: 2,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(3, 1),
                steps: 12,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
        ];
        let available_positions = vec![
            Position::new(1, 0),
            Position::new(2, 0),
            Position::new(2, 1),
            Position::new(3, 0),
            Position::new(3, 1),
        ];
        let result = Runner::get_ready_magic_state(
            &routing_qubits,
            &mut distillation_steps,
            &available_positions,
            &conf,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_get_ready_magic_state() {
        use PatchDirection::*;
        let conf = Configuration {
            width: 4,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };
        let routing_qubits = vec![
            Position::new(1, 0),
            Position::new(2, 0),
            Position::new(2, 1),
        ];
        let mut distillation_steps = vec![
            DistillationState {
                position: Position::new(0, 0),
                steps: 4,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(0, 1),
                steps: 3,
                direction: HorizontalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(0, 2),
                steps: 13,
                direction: HorizontalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(3, 0),
                steps: 2,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(3, 1),
                steps: 12,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(3, 2),
                steps: 13,
                direction: HorizontalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
        ];
        let available_positions = vec![
            Position::new(1, 0),
            Position::new(2, 0),
            Position::new(2, 1),
            Position::new(0, 0),
            Position::new(0, 1),
            Position::new(0, 2),
            Position::new(3, 0),
            Position::new(3, 1),
            Position::new(3, 2),
        ];

        let result = Runner::get_ready_magic_state(
            &routing_qubits,
            &mut distillation_steps,
            &available_positions,
            &conf,
        );

        if let Some((distillation_state, path)) = result {
            assert_eq!(distillation_state.position, Position::new(0, 2));
            assert_eq!(distillation_state.steps, 13);
            assert_eq!(distillation_state.direction, HorizontalZBoundary);
            assert_eq!(path, vec![Position::new(0, 0), Position::new(0, 1)]);
        } else {
            unreachable!("test failed");
        }
    }

    #[test]
    fn test_get_ready_magic_state_prioritizing_long_path_when_cost_matches() {
        use PatchDirection::*;
        let conf = Configuration {
            width: 5,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };
        let routing_qubits = vec![
            Position::new(1, 0),
            Position::new(0, 1),
            Position::new(1, 1),
        ];
        let mut distillation_steps = vec![
            DistillationState {
                position: Position::new(2, 0),
                steps: 4,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(3, 0),
                steps: 13,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(2, 1),
                steps: 2,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(3, 1),
                steps: 2,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(4, 1),
                steps: 13,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
        ];
        let available_positions = vec![
            Position::new(1, 0),
            Position::new(0, 1),
            Position::new(1, 1),
            Position::new(2, 0),
            Position::new(3, 0),
            Position::new(2, 1),
            Position::new(3, 1),
            Position::new(4, 1),
        ];

        let result = Runner::get_ready_magic_state(
            &routing_qubits,
            &mut distillation_steps,
            &available_positions,
            &conf,
        );

        if let Some((distillation_state, path)) = result {
            assert_eq!(distillation_state.position, Position::new(4, 1));
            assert_eq!(distillation_state.steps, 13);
            assert_eq!(distillation_state.direction, VerticalZBoundary);
            assert_eq!(path, vec![Position::new(2, 1), Position::new(3, 1)]);
        } else {
            unreachable!("test failed");
        }
    }

    #[test]
    fn test_get_ready_magic_state_direction_is_taken_into_account() {
        use PatchDirection::*;
        let conf = Configuration {
            width: 5,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };
        let routing_qubits = vec![
            Position::new(1, 0),
            Position::new(0, 1),
            Position::new(1, 1),
        ];
        let mut distillation_steps = vec![
            DistillationState {
                position: Position::new(3, 0),
                steps: 13,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(4, 0),
                steps: 12,
                direction: HorizontalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(2, 1),
                steps: 0,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(3, 1),
                steps: 0,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(4, 1),
                steps: 12,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
        ];
        let available_positions = vec![
            Position::new(1, 0),
            Position::new(0, 1),
            Position::new(1, 1),
            Position::new(3, 0),
            Position::new(4, 0),
            Position::new(2, 1),
            Position::new(3, 1),
            Position::new(4, 1),
        ];

        let result = Runner::get_ready_magic_state(
            &routing_qubits,
            &mut distillation_steps,
            &available_positions,
            &conf,
        );

        if let Some((distillation_state, path)) = result {
            assert_eq!(distillation_state.position, Position::new(3, 0));
            assert_eq!(distillation_state.steps, 13);
            assert_eq!(distillation_state.direction, VerticalZBoundary);
            assert_eq!(
                path,
                vec![
                    Position::new(2, 1),
                    Position::new(3, 1),
                    Position::new(4, 1),
                    Position::new(4, 0)
                ]
            );
        } else {
            unreachable!("test failed");
        }
    }

    #[test]
    fn test_get_ready_magic_state_with_nontrivial_available_positions() {
        use PatchDirection::*;
        let conf = Configuration {
            width: 5,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };
        let routing_qubits = vec![Position::new(1, 0), Position::new(2, 0)];
        let mut distillation_steps = vec![
            DistillationState {
                position: Position::new(0, 0),
                steps: 12,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(0, 1),
                steps: 13,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(3, 0),
                steps: 2,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(3, 1),
                steps: 10,
                direction: VerticalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(4, 0),
                steps: 0,
                direction: HorizontalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(4, 1),
                steps: 12,
                direction: HorizontalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(5, 0),
                steps: 1,
                direction: HorizontalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(5, 1),
                steps: 12,
                direction: HorizontalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
            DistillationState {
                position: Position::new(5, 2),
                steps: 13,
                direction: HorizontalZBoundary,
                is_involved_with_lattice_surgery: false,
            },
        ];
        let available_positions = vec![
            Position::new(2, 0),
            Position::new(0, 1),
            Position::new(3, 0),
            Position::new(3, 1),
            Position::new(4, 1),
            Position::new(5, 0),
            Position::new(5, 1),
            Position::new(5, 2),
        ];

        let result = Runner::get_ready_magic_state(
            &routing_qubits,
            &mut distillation_steps,
            &available_positions,
            &conf,
        );
        if let Some((distillation_state, path)) = result {
            assert_eq!(distillation_state.position, Position::new(5, 2));
            assert_eq!(distillation_state.steps, 13);
            assert_eq!(distillation_state.direction, HorizontalZBoundary);
            assert_eq!(
                path,
                vec![
                    Position::new(3, 0),
                    Position::new(3, 1),
                    Position::new(4, 1),
                    Position::new(5, 1)
                ]
            );
        } else {
            unreachable!("test failed");
        }
    }

    #[test]
    fn test_register_initial_state_for_pi_over_8_rotation_block() {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        use PatchDirection::*;
        let end_cycle = 30_u32;
        let conf = Configuration {
            width: 4,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 13,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 3);
        let id = OperationId::new(0);

        let operations = vec![SingleQubitPiOver8RotationBlock {
            id,
            target: Position::new(0, 0),
            routing_qubits: vec![
                Position::new(1, 0),
                Position::new(0, 1),
                Position::new(1, 1),
            ],
            distillation_qubits: vec![
                Position::new(2, 0),
                Position::new(0, 2),
                Position::new(2, 1),
                Position::new(3, 0),
            ],
            correction_qubits: vec![
                Position::new(2, 1),
                Position::new(1, 0),
                Position::new(0, 1),
            ],
            pi_over_8_axes: vec![Pauli::Z, Pauli::X, Pauli::Z],
            pi_over_4_axes: vec![],
        }];
        let schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        let mut runner = new_runner(operations, schedule, &conf);

        runner.register_initial_state(PiOver8RotationBlock(id));
        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 1);
        let state = runner.pi_over_8_rotation_block_states.get(&id).unwrap();
        match state {
            SingleQubitPiOver8RotationBlockState::PiOver8Rotation {
                index,
                lattice_surgery_steps,
                distillation_steps,
            } => {
                assert_eq!(*index, 0);
                assert_eq!(*lattice_surgery_steps, None);
                assert_eq!(
                    *distillation_steps,
                    vec![
                        DistillationState {
                            position: Position::new(2, 0),
                            steps: 0,
                            direction: VerticalZBoundary,
                            is_involved_with_lattice_surgery: false,
                        },
                        DistillationState {
                            position: Position::new(0, 2),
                            steps: 0,
                            direction: HorizontalZBoundary,
                            is_involved_with_lattice_surgery: false,
                        },
                        DistillationState {
                            position: Position::new(2, 1),
                            steps: 0,
                            direction: VerticalZBoundary,
                            is_involved_with_lattice_surgery: false,
                        },
                        DistillationState {
                            position: Position::new(3, 0),
                            steps: 0,
                            direction: VerticalZBoundary,
                            is_involved_with_lattice_surgery: false,
                        },
                    ]
                );
            }
            _ => unreachable!("test failed"),
        }
    }

    #[test]
    fn test_process_pi_over_8_block_initial_distillation() {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        use PatchDirection::*;
        let end_cycle = 30_u32;
        let conf = Configuration {
            width: 4,
            height: 4,
            code_distance: 3,
            magic_state_distillation_cost: 3,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 3);
        let id = OperationId::new(0);
        let mut rng = RngForTesting::new_with_zero();

        let operations = vec![SingleQubitPiOver8RotationBlock {
            id,
            target: Position::new(0, 0),
            routing_qubits: vec![
                Position::new(1, 0),
                Position::new(0, 1),
                Position::new(1, 1),
            ],
            distillation_qubits: vec![Position::new(2, 0), Position::new(2, 1)],
            correction_qubits: vec![
                Position::new(2, 0),
                Position::new(1, 0),
                Position::new(0, 1),
            ],
            pi_over_8_axes: vec![Pauli::Z, Pauli::X, Pauli::Z],
            pi_over_4_axes: vec![],
        }];
        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 0, 0, 1..10, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 1, 0, 1..10, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 0, 0..10, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 0, 1, 1..10, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 1, 1, 1..10, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 1, 1..10, PiOver8RotationBlock(id));

        let mut runner = new_runner(operations, schedule, &conf);

        runner.register_initial_state(PiOver8RotationBlock(id));
        runner.process_pi_over_8_rotation_block(&mut rng);

        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 1);
        let state = runner.pi_over_8_rotation_block_states.get(&id).unwrap();
        match state {
            SingleQubitPiOver8RotationBlockState::PiOver8Rotation {
                index,
                lattice_surgery_steps,
                distillation_steps,
            } => {
                assert_eq!(*index, 0);
                assert_eq!(*lattice_surgery_steps, None);
                assert_eq!(
                    *distillation_steps,
                    vec![
                        DistillationState {
                            position: Position::new(2, 0),
                            steps: 1,
                            direction: VerticalZBoundary,
                            is_involved_with_lattice_surgery: false,
                        },
                        DistillationState {
                            position: Position::new(2, 1),
                            steps: 0,
                            direction: VerticalZBoundary,
                            is_involved_with_lattice_surgery: false,
                        },
                    ]
                );
            }
            _ => unreachable!("test failed"),
        }

        runner.runtime_cycle += 1;
        runner.process_pi_over_8_rotation_block(&mut rng);

        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 1);
        let state = runner.pi_over_8_rotation_block_states.get(&id).unwrap();
        match state {
            SingleQubitPiOver8RotationBlockState::PiOver8Rotation {
                index,
                lattice_surgery_steps,
                distillation_steps,
            } => {
                assert_eq!(*index, 0);
                assert_eq!(*lattice_surgery_steps, None);
                assert_eq!(
                    *distillation_steps,
                    vec![
                        DistillationState {
                            position: Position::new(2, 0),
                            steps: 2,
                            direction: VerticalZBoundary,
                            is_involved_with_lattice_surgery: false,
                        },
                        DistillationState {
                            position: Position::new(2, 1),
                            steps: 1,
                            direction: VerticalZBoundary,
                            is_involved_with_lattice_surgery: false,
                        },
                    ]
                );
            }
            _ => unreachable!("test failed"),
        }
    }

    #[test]
    fn test_process_pi_over_8_block_without_delay() {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        use PatchDirection::*;
        let end_cycle = 30_u32;
        let conf = Configuration {
            width: 4,
            height: 4,
            code_distance: 3,
            magic_state_distillation_cost: 3,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 2, 3);
        let id = OperationId::new(0);
        let mut rng = RngForTesting::new_with_zero();

        let operations = vec![SingleQubitPiOver8RotationBlock {
            id,
            target: Position::new(0, 0),
            routing_qubits: vec![
                Position::new(1, 0),
                Position::new(0, 1),
                Position::new(1, 1),
            ],
            distillation_qubits: vec![Position::new(2, 0), Position::new(2, 1)],
            correction_qubits: vec![
                Position::new(2, 0),
                Position::new(1, 0),
                Position::new(0, 1),
            ],
            pi_over_8_axes: vec![Pauli::Z, Pauli::X],
            pi_over_4_axes: vec![],
        }];
        let mut schedule = new_occupancy_map(
            conf.width,
            conf.height,
            end_cycle,
            &[mapping.get(q0).unwrap(), mapping.get(q1).unwrap()],
        );
        set_occupancy(&mut schedule, 0, 0, 1..20, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 1, 0, 1..20, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 0, 0..20, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 0, 1, 1..20, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 1, 1, 1..20, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 1, 1..20, PiOver8RotationBlock(id));

        let mut runner = new_runner(operations, schedule, &conf);
        use SingleQubitPiOver8RotationBlockState::Correction as C;
        use SingleQubitPiOver8RotationBlockState::PiOver8Rotation as S;
        let expectation = vec![
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 3,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(1),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 3,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(2),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 3,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(3),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 3,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 1,
                lattice_surgery_steps: Some(1),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                ],
            },
            S {
                index: 1,
                lattice_surgery_steps: Some(2),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                ],
            },
            S {
                index: 1,
                lattice_surgery_steps: Some(3),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 3,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                ],
            },
            C { steps: 1 },
            C { steps: 2 },
            C { steps: 3 },
            C { steps: 4 },
            C { steps: 5 },
            C { steps: 6 },
            C { steps: 7 },
            C { steps: 8 },
            C { steps: 9 },
        ];

        for e in expectation.iter() {
            runner.run_internal_step(&mut rng, false);
            let state = runner.pi_over_8_rotation_block_states.get(&id).unwrap();
            assert_eq!(state, e);
            runner.runtime_cycle += 1;
        }

        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 1);
        runner.perform_pi_over_8_rotation_block_state_transition();
        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 0);
        assert!(runner.removed_operation_ids.contains(&id));
    }

    #[test]
    fn test_process_pi_over_8_block_with_incoming_delay_on_distillation_qubits() {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        use PatchDirection::*;
        let end_cycle = 30_u32;
        let conf = Configuration {
            width: 3,
            height: 3,
            code_distance: 3,
            magic_state_distillation_cost: 2,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        mapping.map(q0, 0, 0);
        let id0 = OperationId::new(0);
        let mut rng = RngForTesting::new_with_zero();

        let operations = vec![SingleQubitPiOver8RotationBlock {
            id: id0,
            target: Position::new(0, 0),
            routing_qubits: vec![
                Position::new(1, 0),
                Position::new(0, 1),
                Position::new(1, 1),
            ],
            distillation_qubits: vec![Position::new(2, 0), Position::new(2, 1)],
            correction_qubits: vec![
                Position::new(2, 0),
                Position::new(1, 0),
                Position::new(0, 1),
            ],
            pi_over_8_axes: vec![Pauli::Z],
            pi_over_4_axes: vec![],
        }];
        let mut schedule =
            new_occupancy_map(conf.width, conf.height, end_cycle, &[mapping.get(q0).unwrap()]);

        set_occupancy(&mut schedule, 0, 0, 5..15, DataQubitInOperation(id0));
        set_occupancy(&mut schedule, 1, 0, 5..15, PiOver8RotationBlock(id0));
        set_occupancy(&mut schedule, 2, 0, 5..15, PiOver8RotationBlock(id0));
        set_occupancy(&mut schedule, 0, 1, 5..15, PiOver8RotationBlock(id0));
        set_occupancy(&mut schedule, 1, 1, 5..15, PiOver8RotationBlock(id0));
        set_occupancy(&mut schedule, 2, 1, 5..15, PiOver8RotationBlock(id0));

        let mut runner = new_runner(operations, schedule, &conf);
        runner.runtime_cycle = 5;

        runner.delay_at[(2, 0)] = 3;
        runner.delay_at[(2, 1)] = 2;

        use SingleQubitPiOver8RotationBlockState::Correction as C;
        use SingleQubitPiOver8RotationBlockState::PiOver8Rotation as S;
        let expectation = vec![
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(1),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(2),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(3),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                ],
            },
            C { steps: 1 },
            C { steps: 2 },
            C { steps: 3 },
            C { steps: 4 },
            C { steps: 5 },
            C { steps: 6 },
            C { steps: 7 },
            C { steps: 8 },
            C { steps: 9 },
        ];

        runner.register_initial_state(PiOver8RotationBlock(id0));
        for e in expectation.iter() {
            runner.perform_pi_over_8_rotation_block_state_transition();
            runner.process_pi_over_8_rotation_block(&mut rng);
            let state = runner.pi_over_8_rotation_block_states.get(&id0).unwrap();
            assert_eq!(state, e);
            runner.runtime_cycle += 1;
        }

        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 1);
        runner.perform_pi_over_8_rotation_block_state_transition();
        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 0);
        assert!(runner.removed_operation_ids.contains(&id0));

        assert_eq!(runner.delay_at[(0, 0)], 3);
        assert_eq!(runner.delay_at[(1, 0)], 6);
        assert_eq!(runner.delay_at[(2, 0)], 3);
        assert_eq!(runner.delay_at[(0, 1)], 6);
        assert_eq!(runner.delay_at[(1, 1)], 3);
        assert_eq!(runner.delay_at[(2, 1)], 0);
        assert_eq!(runner.delay_at[(0, 2)], 0);
        assert_eq!(runner.delay_at[(1, 2)], 0);
        assert_eq!(runner.delay_at[(2, 2)], 0);
    }

    #[test]
    fn test_process_pi_over_8_block_with_incoming_delay_on_target_qubit() {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        use PatchDirection::*;
        let end_cycle = 30_u32;
        let conf = Configuration {
            width: 3,
            height: 3,
            code_distance: 3,
            magic_state_distillation_cost: 2,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        mapping.map(q0, 0, 0);
        let id = OperationId::new(0);
        let mut rng = RngForTesting::new_with_zero();

        let operations = vec![SingleQubitPiOver8RotationBlock {
            id,
            target: Position::new(0, 0),
            routing_qubits: vec![
                Position::new(1, 0),
                Position::new(0, 1),
                Position::new(1, 1),
            ],
            distillation_qubits: vec![Position::new(2, 0), Position::new(2, 1)],
            correction_qubits: vec![
                Position::new(2, 0),
                Position::new(1, 0),
                Position::new(0, 1),
            ],
            pi_over_8_axes: vec![Pauli::Z],
            pi_over_4_axes: vec![],
        }];
        let mut schedule =
            new_occupancy_map(conf.width, conf.height, end_cycle, &[mapping.get(q0).unwrap()]);

        set_occupancy(&mut schedule, 0, 0, 5..15, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 1, 0, 5..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 0, 5..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 0, 1, 5..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 1, 1, 5..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 1, 5..15, PiOver8RotationBlock(id));

        let mut runner = new_runner(operations, schedule, &conf);
        runner.runtime_cycle = 5;

        runner.delay_at[(0, 0)] = 4;

        use SingleQubitPiOver8RotationBlockState::Correction as C;
        use SingleQubitPiOver8RotationBlockState::PiOver8Rotation as S;
        let expectation = vec![
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(1),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(2),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(3),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            C { steps: 1 },
            C { steps: 2 },
            C { steps: 3 },
            C { steps: 4 },
            C { steps: 5 },
            C { steps: 6 },
            C { steps: 7 },
            C { steps: 8 },
            C { steps: 9 },
        ];

        runner.register_initial_state(PiOver8RotationBlock(id));
        for e in expectation.iter() {
            runner.perform_pi_over_8_rotation_block_state_transition();
            runner.process_pi_over_8_rotation_block(&mut rng);
            let state = runner.pi_over_8_rotation_block_states.get(&id).unwrap();
            assert_eq!(state, e);
            runner.runtime_cycle += 1;
        }

        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 1);
        runner.perform_pi_over_8_rotation_block_state_transition();
        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 0);
        assert!(runner.removed_operation_ids.contains(&id));

        assert_eq!(runner.delay_at[(0, 0)], 3);
        assert_eq!(runner.delay_at[(1, 0)], 6);
        assert_eq!(runner.delay_at[(2, 0)], 3);
        assert_eq!(runner.delay_at[(0, 1)], 6);
        assert_eq!(runner.delay_at[(1, 1)], 3);
        assert_eq!(runner.delay_at[(2, 1)], 0);
        assert_eq!(runner.delay_at[(0, 2)], 0);
        assert_eq!(runner.delay_at[(1, 2)], 0);
        assert_eq!(runner.delay_at[(2, 2)], 0);
    }

    #[test]
    fn test_overlapping_process_pi_over_8_blocks() {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        let end_cycle = 65_u32;
        let conf = Configuration {
            width: 4,
            height: 3,
            code_distance: 3,
            magic_state_distillation_cost: 2,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        mapping.map(q0, 0, 0);
        mapping.map(q1, 3, 2);
        let id0 = OperationId::new(0);
        let id1 = OperationId::new(1);
        let mut rng = RngForTesting::new_with_zero();

        let operations = vec![
            SingleQubitPiOver8RotationBlock {
                id: id0,
                target: Position::new(0, 0),
                routing_qubits: vec![
                    Position::new(1, 0),
                    Position::new(0, 1),
                    Position::new(1, 1),
                ],
                distillation_qubits: vec![Position::new(2, 0)],
                correction_qubits: vec![
                    Position::new(2, 0),
                    Position::new(1, 0),
                    Position::new(0, 1),
                ],
                pi_over_8_axes: vec![Pauli::Z],
                pi_over_4_axes: vec![],
            },
            SingleQubitPiOver8RotationBlock {
                id: id1,
                target: Position::new(3, 2),
                routing_qubits: vec![
                    Position::new(3, 1),
                    Position::new(2, 1),
                    Position::new(2, 2),
                ],
                distillation_qubits: vec![
                    Position::new(3, 0),
                    Position::new(1, 1),
                    Position::new(0, 1),
                    Position::new(1, 0),
                ],
                correction_qubits: vec![
                    Position::new(3, 0),
                    Position::new(3, 1),
                    Position::new(2, 2),
                ],
                pi_over_8_axes: vec![Pauli::Z],
                pi_over_4_axes: vec![],
            },
        ];

        let mut schedule =
            new_occupancy_map(conf.width, conf.height, end_cycle, &[mapping.get(q0).unwrap()]);

        set_occupancy(&mut schedule, 0, 0, 45..55, DataQubitInOperation(id0));
        set_occupancy(&mut schedule, 1, 0, 45..55, PiOver8RotationBlock(id0));
        set_occupancy(&mut schedule, 2, 0, 45..55, PiOver8RotationBlock(id0));
        set_occupancy(&mut schedule, 0, 1, 45..55, PiOver8RotationBlock(id0));
        set_occupancy(&mut schedule, 1, 1, 45..55, PiOver8RotationBlock(id0));

        set_occupancy(&mut schedule, 3, 2, 55..65, DataQubitInOperation(id1));
        set_occupancy(&mut schedule, 1, 0, 55..65, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 3, 0, 55..65, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 0, 1, 55..65, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 1, 1, 55..65, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 2, 1, 55..65, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 3, 1, 55..65, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 2, 2, 55..65, PiOver8RotationBlock(id1));

        let mut runner = new_runner(operations, schedule, &conf);
        runner.delay_at[(0, 0)] = 45;
        runner.runtime_cycle = 45;

        while runner.removed_operation_ids.len() < 2 {
            runner.run_internal_step(&mut rng, true);
            runner.runtime_cycle += 1;

            // This assertion is flaky because it depends on the iteration of of
            // `runner.pi_over_8_rotation_block_states` which is not deterministic.
            assert!(runner.runtime_cycle < 1000);
        }

        assert_eq!(runner.runtime_cycle, 103);
    }

    #[test]
    fn test_process_pi_over_8_block_with_incoming_delay_on_routing_qubit() {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        use PatchDirection::*;
        let end_cycle = 30_u32;
        let conf = Configuration {
            width: 3,
            height: 3,
            code_distance: 3,
            magic_state_distillation_cost: 2,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        mapping.map(q0, 0, 0);
        let id = OperationId::new(0);
        let mut rng = RngForTesting::new_with_zero();

        let operations = vec![SingleQubitPiOver8RotationBlock {
            id,
            target: Position::new(0, 0),
            routing_qubits: vec![
                Position::new(1, 0),
                Position::new(0, 1),
                Position::new(1, 1),
            ],
            distillation_qubits: vec![Position::new(2, 0), Position::new(2, 1)],
            correction_qubits: vec![
                Position::new(2, 0),
                Position::new(1, 0),
                Position::new(0, 1),
            ],
            pi_over_8_axes: vec![Pauli::Z],
            pi_over_4_axes: vec![],
        }];
        let mut schedule =
            new_occupancy_map(conf.width, conf.height, end_cycle, &[mapping.get(q0).unwrap()]);

        set_occupancy(&mut schedule, 0, 0, 5..15, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 1, 0, 5..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 0, 5..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 0, 1, 5..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 1, 1, 5..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 1, 5..15, PiOver8RotationBlock(id));

        let mut runner = new_runner(operations, schedule, &conf);
        runner.runtime_cycle = 5;

        runner.delay_at[(0, 1)] = 4;

        use SingleQubitPiOver8RotationBlockState::Correction as C;
        use SingleQubitPiOver8RotationBlockState::PiOver8Rotation as S;
        let expectation = vec![
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(1),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(2),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                index: 0,
                lattice_surgery_steps: Some(3),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            C { steps: 1 },
            C { steps: 2 },
            C { steps: 3 },
            C { steps: 4 },
            C { steps: 5 },
            C { steps: 6 },
            C { steps: 7 },
            C { steps: 8 },
            C { steps: 9 },
        ];

        for e in expectation.iter() {
            runner.run_internal_step(&mut rng, true);
            let state = runner.pi_over_8_rotation_block_states.get(&id).unwrap();
            assert_eq!(state, e);
            runner.runtime_cycle += 1;
        }

        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 1);
        runner.perform_pi_over_8_rotation_block_state_transition();
        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 0);
        assert!(runner.removed_operation_ids.contains(&id));

        assert_eq!(runner.delay_at[(0, 0)], 3);
        assert_eq!(runner.delay_at[(1, 0)], 6);
        assert_eq!(runner.delay_at[(2, 0)], 3);
        assert_eq!(runner.delay_at[(0, 1)], 6);
        assert_eq!(runner.delay_at[(1, 1)], 3);
        assert_eq!(runner.delay_at[(2, 1)], 0);
        assert_eq!(runner.delay_at[(0, 2)], 0);
        assert_eq!(runner.delay_at[(1, 2)], 0);
        assert_eq!(runner.delay_at[(2, 2)], 0);
    }

    #[test]
    fn test_process_pi_over_8_block_with_distillation_failures() {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        use PatchDirection::*;
        let end_cycle = 30_u32;
        let conf = Configuration {
            width: 4,
            height: 4,
            code_distance: 3,
            magic_state_distillation_cost: 2,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        mapping.map(q0, 0, 0);
        let prev_id = OperationId::new(0);
        let id = OperationId::new(1);
        let mut rng = RngForTesting::new(&[u64::MAX, u64::MAX, 0, 0, u64::MAX]);

        let operations = vec![
            PiOver4Rotation {
                id: prev_id,
                targets: vec![(Position::new(0, 0), Pauli::Z)],
                ancilla_qubits: vec![Position::new(0, 1), Position::new(1, 1)],
            },
            SingleQubitPiOver8RotationBlock {
                id,
                target: Position::new(0, 0),
                routing_qubits: vec![
                    Position::new(1, 0),
                    Position::new(0, 1),
                    Position::new(1, 1),
                ],
                distillation_qubits: vec![Position::new(2, 0), Position::new(2, 1)],
                correction_qubits: vec![
                    Position::new(2, 0),
                    Position::new(1, 0),
                    Position::new(0, 1),
                ],
                pi_over_8_axes: vec![Pauli::Z, Pauli::Y],
                pi_over_4_axes: vec![],
            },
        ];
        let qubit_positions = [mapping.get(q0).unwrap()];
        let mut schedule = new_occupancy_map(conf.width, conf.height, end_cycle, &qubit_positions);

        set_occupancy(&mut schedule, 0, 0, 0..3, DataQubitInOperation(prev_id));
        set_occupancy(&mut schedule, 0, 1, 0..3, LatticeSurgery(prev_id));
        set_occupancy(&mut schedule, 1, 1, 0..3, LatticeSurgery(prev_id));

        set_occupancy(&mut schedule, 0, 0, 3..15, DataQubitInOperation(id));
        set_occupancy(&mut schedule, 1, 0, 3..18, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 0, 0..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 0, 1, 3..18, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 1, 1, 3..15, PiOver8RotationBlock(id));
        set_occupancy(&mut schedule, 2, 1, 1..9, PiOver8RotationBlock(id));

        let mut runner = new_runner(operations, schedule, &conf);

        use SingleQubitPiOver8RotationBlockState::Correction as C;
        use SingleQubitPiOver8RotationBlockState::PiOver8Rotation as S;
        let expectation = vec![
            S {
                // runtime_clock = 0
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                // runtime_clock = 1
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0, // <= distillation failure
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                // runtime_clock = 2
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0, // <= distillation failure
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                // runtime_clock = 3
                index: 0,
                lattice_surgery_steps: None,
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 2, // <= distillation success
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                // runtime_clock = 4
                index: 0,
                lattice_surgery_steps: Some(1),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2, // <= distillation success
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                // runtime_clock = 5
                index: 0,
                lattice_surgery_steps: Some(2),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                // runtime_clock = 6
                index: 0,
                lattice_surgery_steps: Some(3),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 2,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                ],
            },
            S {
                // runtime_clock = 7
                index: 1,
                lattice_surgery_steps: Some(1),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                ],
            },
            S {
                // runtime_clock = 8
                index: 1,
                lattice_surgery_steps: Some(2),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 0, // <= distillation failure
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                ],
            },
            S {
                // runtime_clock = 9
                index: 1,
                lattice_surgery_steps: Some(3),
                distillation_steps: vec![
                    DistillationState {
                        position: Position::new(2, 0),
                        steps: 1,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: false,
                    },
                    DistillationState {
                        position: Position::new(2, 1),
                        steps: 0,
                        direction: VerticalZBoundary,
                        is_involved_with_lattice_surgery: true,
                    },
                ],
            },
            C { steps: 1 },
            C { steps: 2 },
            C { steps: 3 },
            C { steps: 4 },
            C { steps: 5 },
            C { steps: 6 },
            C { steps: 7 },
            C { steps: 8 },
            C { steps: 9 },
        ];

        for e in expectation.iter() {
            runner.run_internal_step(&mut rng, true);
            let state = runner.pi_over_8_rotation_block_states.get(&id).unwrap();
            assert_eq!(state, e);

            runner.runtime_cycle += 1;
        }

        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 1);
        runner.perform_pi_over_8_rotation_block_state_transition();
        assert_eq!(runner.pi_over_8_rotation_block_states.len(), 0);
        assert!(runner.removed_operation_ids.contains(&id));

        assert_eq!(runner.delay_at[(0, 0)], 1);
        assert_eq!(runner.delay_at[(1, 0)], 1);
        assert_eq!(runner.delay_at[(2, 0)], 1);
        assert_eq!(runner.delay_at[(0, 1)], 1);
        assert_eq!(runner.delay_at[(1, 1)], 1);
        assert_eq!(runner.delay_at[(2, 1)], 1);
        assert_eq!(runner.delay_at[(0, 2)], 0);
        assert_eq!(runner.delay_at[(1, 2)], 0);
        assert_eq!(runner.delay_at[(2, 2)], 0);
    }

    #[test]
    fn test_process_pi_over_8_block_without_suppressing_delay_reduction() {
        use BoardOccupancy::*;
        use OperationWithAdditionalData::*;
        let end_cycle = 30_u32;
        let conf = Configuration {
            width: 3,
            height: 4,
            code_distance: 5,
            magic_state_distillation_cost: 3,
            magic_state_distillation_success_rate: 0.5,
            ..default_conf()
        };

        let mut mapping = DataQubitMapping::new(conf.width, conf.height);
        let q0 = Qubit::new(0);
        mapping.map(q0, 0, 0);
        let id1 = OperationId::new(1);
        let id2 = OperationId::new(2);
        let id3 = OperationId::new(3);
        let mut rng = RngForTesting::new(&[u64::MAX, u64::MAX, 0, u64::MAX]);

        let operations = vec![
            SingleQubitPiOver8RotationBlock {
                id: id1,
                target: Position::new(0, 0),
                routing_qubits: vec![
                    Position::new(1, 0),
                    Position::new(0, 1),
                    Position::new(1, 1),
                ],
                distillation_qubits: vec![Position::new(2, 0), Position::new(2, 1)],
                correction_qubits: vec![
                    Position::new(2, 0),
                    Position::new(1, 0),
                    Position::new(0, 1),
                ],
                pi_over_8_axes: vec![Pauli::Z, Pauli::Y],
                pi_over_4_axes: vec![],
            },
            SingleQubitPiOver8RotationBlock {
                id: id2,
                target: Position::new(0, 0),
                routing_qubits: vec![
                    Position::new(1, 0),
                    Position::new(0, 1),
                    Position::new(1, 1),
                ],
                distillation_qubits: vec![Position::new(0, 2), Position::new(1, 2)],
                correction_qubits: vec![
                    Position::new(0, 2),
                    Position::new(1, 0),
                    Position::new(0, 1),
                ],
                pi_over_8_axes: vec![Pauli::Z, Pauli::Y],
                pi_over_4_axes: vec![],
            },
            SingleQubitPiOver8RotationBlock {
                id: id3,
                target: Position::new(0, 0),
                routing_qubits: vec![
                    Position::new(1, 0),
                    Position::new(0, 1),
                    Position::new(1, 1),
                ],
                distillation_qubits: vec![Position::new(0, 2), Position::new(0, 3)],
                correction_qubits: vec![
                    Position::new(0, 2),
                    Position::new(1, 0),
                    Position::new(0, 1),
                ],
                pi_over_8_axes: vec![Pauli::Z, Pauli::Y],
                pi_over_4_axes: vec![],
            },
        ];
        let qubit_positions = [mapping.get(q0).unwrap()];
        let mut schedule = new_occupancy_map(conf.width, conf.height, end_cycle, &qubit_positions);

        set_occupancy(&mut schedule, 0, 0, 0..9, DataQubitInOperation(id1));
        set_occupancy(&mut schedule, 1, 0, 0..9, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 0, 1, 0..9, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 1, 1, 0..9, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 2, 0, 0..9, PiOver8RotationBlock(id1));
        set_occupancy(&mut schedule, 2, 1, 0..9, PiOver8RotationBlock(id1));

        set_occupancy(&mut schedule, 0, 0, 10..19, DataQubitInOperation(id2));
        set_occupancy(&mut schedule, 1, 0, 10..19, PiOver8RotationBlock(id2));
        set_occupancy(&mut schedule, 0, 1, 10..19, PiOver8RotationBlock(id2));
        set_occupancy(&mut schedule, 1, 1, 10..19, PiOver8RotationBlock(id2));
        set_occupancy(&mut schedule, 0, 2, 10..19, PiOver8RotationBlock(id2));
        set_occupancy(&mut schedule, 1, 2, 10..19, PiOver8RotationBlock(id2));

        set_occupancy(&mut schedule, 0, 0, 20..30, DataQubitInOperation(id3));
        set_occupancy(&mut schedule, 1, 0, 20..30, PiOver8RotationBlock(id3));
        set_occupancy(&mut schedule, 0, 1, 20..30, PiOver8RotationBlock(id3));
        set_occupancy(&mut schedule, 1, 1, 20..30, PiOver8RotationBlock(id3));
        set_occupancy(&mut schedule, 0, 2, 20..30, PiOver8RotationBlock(id3));
        set_occupancy(&mut schedule, 0, 3, 20..30, PiOver8RotationBlock(id3));

        let mut runner = new_runner(operations, schedule, &conf);

        while runner.removed_operation_ids.len() < 3 {
            runner.run_internal_step(&mut rng, false);
            runner.runtime_cycle += 1;
        }

        assert_eq!(runner.delay_at[(0, 0)], 57);
        assert_eq!(runner.delay_at[(1, 0)], 61);
        assert_eq!(runner.delay_at[(2, 0)], 26);
        assert_eq!(runner.delay_at[(0, 1)], 61);
        assert_eq!(runner.delay_at[(1, 1)], 57);
        assert_eq!(runner.delay_at[(2, 1)], 17);
        assert_eq!(runner.delay_at[(0, 2)], 56);
        assert_eq!(runner.delay_at[(1, 2)], 31);
        assert_eq!(runner.delay_at[(2, 2)], 0);
        assert_eq!(runner.delay_at[(0, 3)], 47);
        assert_eq!(runner.delay_at[(1, 3)], 0);
        assert_eq!(runner.delay_at[(2, 3)], 0);
    }
}
