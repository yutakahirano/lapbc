use std::collections::{HashMap, HashSet};
use std::ops::{Index, IndexMut};

use rand::Rng;

use crate::board::Board;
use crate::board::BoardOccupancy;
use crate::board::Configuration;
use crate::board::Map2D;
use crate::board::OperationId;

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

pub struct Runner {
    schedule: OccupancyMap,
    conf: Configuration,
}

impl Runner {
    pub fn new(board: &Board) -> Self {
        Runner {
            schedule: OccupancyMap::new(board),
            conf: board.configuration().clone(),
        }
    }

    // Runs the schedule, and returns the number of cycles it took to complete.
    pub fn run(&self) -> u32 {
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
                        BoardOccupancy::LatticeSurgery(id) => id,
                        BoardOccupancy::DataQubitInOperation(id) => id,
                        BoardOccupancy::MagicStateDistillation { .. } => {
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
                        BoardOccupancy::LatticeSurgery(id) => id,
                        BoardOccupancy::DataQubitInOperation(id) => id,
                        _ => continue,
                    };
                    delay_map[(x, y)] = *delay_with_operation_ids.get(&id).unwrap();
                }
            }

            // Decrease the delay on qubits that are idle.
            for y in 0..height {
                for x in 0..width {
                    let occupancy = &self.schedule[(x, y, scheduled_cycle)];
                    if *occupancy == BoardOccupancy::Vacant
                        || *occupancy == BoardOccupancy::IdleDataQubit
                    {
                        delay_map[(x, y)] = delay_map[(x, y)].saturating_sub(1);
                    }
                }
            }

            // Deal with magic state distillation that can generate new delay.
            for distillation in ending_distillations {
                if let BoardOccupancy::MagicStateDistillation {
                    id: _,
                    num_distillations,
                    num_distillations_on_retry,
                } = distillation
                {
                    let mut new_delay = 0;
                    let mut rnd = rand::thread_rng();
                    let mut done = false;
                    for _ in 0..num_distillations {
                        if rnd.gen_range(0.0..1.0) < single_success_rate {
                            done = true;
                            break;
                        }
                    }

                    while !done {
                        new_delay += distillation_cost;
                        for _ in 0..num_distillations_on_retry {
                            if rnd.gen_range(0.0..1.0) < single_success_rate {
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
}
