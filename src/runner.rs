use std::collections::{HashMap, HashSet};
use std::ops::{Index, IndexMut};

use rand::{thread_rng, Rng};

use crate::board::Configuration;
use crate::board::Map2D;
use crate::board::OperationId;
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

enum Blocking {
    // No blocking is required.
    None,
    // All magic state distillation operations failed, and we need to retry them.
    DistillationFailure,
    // A lattice surgery operation is blocked by arbitrary operation running on any of the qubits
    // on which it is performed.
    LatticeSurgeryIsWaiting,
}

pub struct Runner {
    operations: HashMap<OperationId, OperationWithAdditionalData>,
    schedule: OccupancyMap,
    conf: Configuration,
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
        Runner {
            operations: board
                .operations()
                .iter()
                .map(|op| (op.id(), op.clone()))
                .collect(),
            schedule,
            conf: board.configuration().clone(),
        }
    }

    // Runs the schedule, and returns the number of cycles it took to complete.
    fn run_internal<R: Rng>(&self, rng: &mut R) -> u32 {
        use BoardOccupancy::*;
        let width = self.schedule.width;
        let height = self.schedule.height;
        let single_success_rate = self.conf.magic_state_distillation_success_rate;
        let distillation_cost = self.conf.magic_state_distillation_cost;
        let mut delay_map = Map2D::<u32>::new_with_value(width, height, 0);
        for scheduled_cycle in 0..self.schedule.end_cycle() {
            // Ensure that qubits involved in a lattice surgery operation have the same delay.
            let mut delay_with_operation_ids: HashMap<OperationId, u32> = HashMap::new();
            let mut ending_distillations: HashSet<BoardOccupancy> = HashSet::new();
            for y in 0..height {
                for x in 0..width {
                    let occupancy = &self.schedule[(x, y, scheduled_cycle)];
                    let id = match *occupancy {
                        LatticeSurgery(id) => id,
                        DataQubitInOperation(id) => id,
                        MagicStateDistillation { .. } => {
                            // Here we assume that each distillation block ends at the same cycle.
                            if *occupancy != self.schedule[(x, y, scheduled_cycle + 1)] {
                                ending_distillations.insert(occupancy.clone());
                            }
                            continue;
                        }
                        _ => continue,
                    };

                    let delay = delay_with_operation_ids.entry(id).or_insert(0);
                    *delay = std::cmp::max(*delay, delay_map[(x, y)]);
                }
            }
            for y in 0..height {
                for x in 0..width {
                    let occupancy = &self.schedule[(x, y, scheduled_cycle)];
                    let id = match *occupancy {
                        LatticeSurgery(id) => id,
                        DataQubitInOperation(id) => id,
                        _ => continue,
                    };
                    delay_map[(x, y)] = *delay_with_operation_ids.get(&id).unwrap();
                }
            }

            // Decrease the delay on qubits that are idle.
            for y in 0..height {
                for x in 0..width {
                    let occupancy = &self.schedule[(x, y, scheduled_cycle)];
                    if occupancy.is_vacant_or_idle() {
                        delay_map[(x, y)] = delay_map[(x, y)].saturating_sub(1);
                    }
                }
            }

            // Deal with magic state distillation that can generate new delay.
            for distillation in ending_distillations {
                if let BoardOccupancy::MagicStateDistillation(id) = distillation {
                    let op = self.operations.get(&id).unwrap();
                    let (num_distillations, num_distillations_on_retry) =
                        if let OperationWithAdditionalData::PiOver8Rotation {
                            num_distillations,
                            num_distillations_on_retry,
                            ..
                        } = op
                        {
                            (*num_distillations, *num_distillations_on_retry)
                        } else {
                            unreachable!();
                        };

                    let mut new_delay = 0;
                    let mut done = false;
                    for _ in 0..num_distillations {
                        if rng.gen_range(0.0..1.0) < single_success_rate {
                            done = true;
                            break;
                        }
                    }

                    while !done {
                        new_delay += distillation_cost;
                        for _ in 0..num_distillations_on_retry {
                            if rng.gen_range(0.0..1.0) < single_success_rate {
                                done = true;
                                break;
                            }
                        }
                    }
                    // Here we do not care which distillation succeeded. That is not problematic
                    // because our scheduler schedules a lattice surgery operation that covers
                    // all the qubits right after distillation.
                    for y in 0..height {
                        for x in 0..width {
                            if self.schedule[(x, y, scheduled_cycle)] == distillation {
                                delay_map[(x, y)] += new_delay;
                            }
                        }
                    }
                }
            }
        }

        let mut delay = 0;
        for y in 0..height {
            for x in 0..width {
                delay = std::cmp::max(delay, delay_map[(x, y)]);
            }
        }
        delay
    }

    pub fn run(&self) -> u32 {
        let mut rng = thread_rng();
        self.run_internal(&mut rng)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::{
        board::Position,
        mapping::{DataQubitMapping, Qubit},
        pbc::Pauli,
    };

    use super::*;

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
        let runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_only_clifford() {
        use BoardOccupancy::*;
        let end_cycle = 8_u32;
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
        set_occupancy(&mut schedule, 0, 1, 5..8, YMeasurement(id));

        let runner = new_runner(operations, schedule, &conf);
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

        let runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 0);
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

        let runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 13);
        assert_eq!(rng.counter, 5);
    }

    #[test]
    fn test_magic_state_distillation_retry() {
        use BoardOccupancy::*;
        let end_cycle = 35_u32;
        let mut rng = RngForTesting::new(&[u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX, 0]);
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

        let runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 26);
        assert_eq!(rng.counter, 6);
    }

    #[test]
    fn test_delay_propagation() {
        use BoardOccupancy::*;
        let end_cycle = 49_u32;
        let mut rng = RngForTesting::new(&[
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            0,
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
                ancilla_qubits: vec![p(1, 1), p(2, 1), p(2, 2)],
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

        let runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 26);
        assert_eq!(rng.counter, 9);
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

        let runner = new_runner(operations, schedule, &conf);
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

        let runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 11);
        assert_eq!(rng.counter, 5);
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

        let runner = new_runner(operations, schedule, &conf);
        let result = runner.run_internal(&mut rng);
        assert_eq!(result, 11);
        assert_eq!(rng.counter, 2);
    }
}
