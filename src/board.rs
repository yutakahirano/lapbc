use std::collections::{HashSet, VecDeque};
use std::ops::{Index, IndexMut};
use std::{collections::HashMap, ops::Range};

use serde::ser::{SerializeSeq, SerializeStruct};
use serde::Serialize;

use crate::mapping::{DataQubitMapping, Qubit};
use crate::pbc::{Angle, Operation, Pauli, PauliRotation};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OperationId {
    id: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialOrd, PartialEq, Serialize)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}

impl Position {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

struct Targets<'a> {
    internal: &'a [(Position, Pauli)],
}

impl<'a> Targets<'a> {
    fn new(targets: &'a [(Position, Pauli)]) -> Self {
        Self { internal: targets }
    }
}

impl serde::Serialize for Targets<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let seq = self.internal;
        let mut state = serializer.serialize_seq(Some(seq.len()))?;

        #[derive(Serialize)]
        struct Element {
            x: u32,
            y: u32,
            axis: Pauli,
        }

        for (pos, pauli) in seq {
            state.serialize_element(&Element {
                x: pos.x,
                y: pos.y,
                axis: *pauli,
            })?;
        }

        state.end()
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum OperationWithAdditionalData {
    PiOver4Rotation {
        id: OperationId,
        targets: Vec<(Position, Pauli)>,
        ancilla_qubits: Vec<Position>,
    },
    PiOver8Rotation {
        id: OperationId,
        targets: Vec<(Position, Pauli)>,
        routing_qubits: Vec<Position>,
        distillation_qubits: Vec<Position>,
        num_distillations: u32,
        num_distillations_on_retry: u32,
    },
    SingleQubitPiOver8RotationBlock {
        id: OperationId,
        target: Position,
        routing_qubits: Vec<Position>,
        distillation_qubits: Vec<Position>,
        correction_qubits: Vec<Position>,
        pi_over_8_axes: Vec<Pauli>,
        pi_over_4_axes: Vec<Pauli>,
    }, // Measurement should be listed below, but it is not implemented yet.
}

impl OperationWithAdditionalData {
    pub fn id(&self) -> OperationId {
        use OperationWithAdditionalData::*;
        match self {
            PiOver4Rotation { id, .. } => *id,
            PiOver8Rotation { id, .. } => *id,
            SingleQubitPiOver8RotationBlock { id, .. } => *id,
        }
    }
}

impl serde::Serialize for OperationWithAdditionalData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use OperationWithAdditionalData::*;
        macro_rules! put {
            ( $type:expr, $length:expr, $($x:expr ),* ) => {{
                let mut state = serializer.serialize_struct("Operation", $length + 1)?;
                state.serialize_field("type", $type)?;
                $(
                    let (k, v) = $x;
                    state.serialize_field(k, &v)?;
                )*
                state.end()
            }};
        }

        match self {
            PiOver4Rotation {
                id,
                targets,
                ancilla_qubits,
            } => {
                put!(
                    "PI_OVER_4_ROTATION",
                    3,
                    ("id", id.id),
                    ("targets", Targets::new(targets)),
                    ("ancilla_qubits", ancilla_qubits)
                )
            }
            PiOver8Rotation {
                id,
                targets,
                routing_qubits,
                distillation_qubits,
                num_distillations,
                num_distillations_on_retry,
            } => {
                put!(
                    "PI_OVER_8_ROTATION",
                    6,
                    ("id", id.id),
                    ("targets", Targets::new(targets)),
                    ("routing_qubits", routing_qubits),
                    ("distillation_qubits", distillation_qubits),
                    ("num_distillations", num_distillations),
                    ("num_distillations_on_retry", num_distillations_on_retry)
                )
            }
            SingleQubitPiOver8RotationBlock {
                id,
                target,
                routing_qubits,
                distillation_qubits,
                correction_qubits,
                pi_over_8_axes,
                pi_over_4_axes,
            } => {
                put!(
                    "SINGLE_QUBIT_PI_OVER_8_ROTATION_BLOCK",
                    7,
                    ("id", id.id),
                    ("target", target),
                    ("routing_qubits", routing_qubits),
                    ("distillation_qubits", distillation_qubits),
                    ("correction_qubits", correction_qubits),
                    ("pi_over_8_axes", pi_over_8_axes),
                    ("pi_over_4_axes", pi_over_4_axes)
                )
            }
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum BoardOccupancy {
    Vacant,
    LatticeSurgery(OperationId),
    IdleDataQubit,
    DataQubitInOperation(OperationId),
    YInitialization(OperationId),
    YMeasurement(OperationId),
    MagicStateDistillation(OperationId),
    PiOver8RotationBlock(OperationId),
}

impl BoardOccupancy {
    pub fn is_vacant_or_idle(&self) -> bool {
        matches!(self, BoardOccupancy::Vacant | BoardOccupancy::IdleDataQubit)
    }

    pub fn operation_id(&self) -> Option<OperationId> {
        use BoardOccupancy::*;
        match self {
            Vacant => None,
            LatticeSurgery(id) => Some(*id),
            IdleDataQubit => None,
            DataQubitInOperation(id) => Some(*id),
            YInitialization(id) => Some(*id),
            YMeasurement(id) => Some(*id),
            MagicStateDistillation(id) => Some(*id),
            PiOver8RotationBlock(id) => Some(*id),
        }
    }
}

impl serde::Serialize for BoardOccupancy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use BoardOccupancy::*;

        let type_: &str;
        let mut operation_id: Option<OperationId> = None;
        match self {
            Vacant => {
                type_ = "VACANT";
            }
            LatticeSurgery(id) => {
                type_ = "LATTICE_SURGERY";
                operation_id = Some(*id);
            }
            IdleDataQubit => {
                type_ = "IDLE_DATA_QUBIT";
            }
            DataQubitInOperation(id) => {
                type_ = "DATA_QUBIT_IN_OPERATION";
                operation_id = Some(*id);
            }
            YInitialization(id) => {
                type_ = "Y_INITIALIZATION";
                operation_id = Some(*id);
            }
            YMeasurement(id) => {
                type_ = "Y_MEASUREMENT";
                operation_id = Some(*id);
            }
            MagicStateDistillation(id) => {
                type_ = "MAGIC_STATE_DISTILLATION";
                operation_id = Some(*id);
            }
            PiOver8RotationBlock(id) => {
                type_ = "PI_OVER_8_ROTATION_BLOCK";
                operation_id = Some(*id);
            }
        }
        let len = if operation_id.is_some() { 2 } else { 1 };
        let mut state = serializer.serialize_struct("ScheduleEntry", len)?;
        state.serialize_field("type", type_)?;
        if let Some(id) = operation_id {
            state.serialize_field("operation_id", &id.id)?;
        }
        state.end()
    }
}

pub struct Board {
    conf: Configuration,

    data_qubit_mapping: HashMap<Qubit, (u32, u32)>,
    occupancy: Vec<BoardOccupancy>,
    cycle_after_last_operation_at: Vec<u32>,
    cycle: u32,
    current_operation_id: OperationId,
    operations: Vec<OperationWithAdditionalData>,
    arbitrary_angle_rotation_map: Vec<(f64, Vec<Pauli>, Vec<Pauli>)>,
}

#[derive(Clone, Debug)]
pub struct Configuration {
    pub width: u32,
    pub height: u32,
    // The code distance.
    pub code_distance: u32,
    // The cycle needed to distill a magic state.
    pub magic_state_distillation_cost: u32,
    // The number of distillations needed for each pi/8 rotation.
    pub num_distillations_for_pi_over_8_rotation: u32,
    // The probability to distill a magic state successfully.
    pub magic_state_distillation_success_rate: f64,

    pub num_distillations_for_pi_over_8_rotation_block: u32,
    pub single_qubit_pi_over_8_rotation_block_depth_ratio: f64,
    pub single_qubit_arbitrary_angle_rotation_precision: f64,
    pub preferable_distillation_area_size: u32,

    pub enable_two_qubit_pi_over_4_rotation_with_y_initialization: bool,
}

#[derive(Clone, Debug)]
pub struct Map2D<T: Clone> {
    width: u32,
    height: u32,
    map: Vec<T>,
}

struct AncillaAvailability {
    map: Map2D<bool>,
}

impl OperationId {
    #[allow(dead_code)]
    pub fn new(id: u32) -> Self {
        Self { id }
    }

    fn increment(&mut self) {
        self.id += 1;
    }
}

pub fn y_initialization_cost(distance: u32) -> u32 {
    distance + y_measurement_cost(distance)
}

pub fn y_measurement_cost(distance: u32) -> u32 {
    (distance / 2) + 2
}

impl Board {
    pub fn new(mapping: DataQubitMapping, conf: &Configuration) -> Self {
        assert_eq!(mapping.width, conf.width);
        assert_eq!(mapping.height, conf.height);

        let occupancy = vec![];
        let current_operation_id = OperationId { id: 0 };

        Self {
            conf: conf.clone(),
            data_qubit_mapping: mapping.iter().map(|&(x, y, q)| (q, (x, y))).collect(),
            occupancy,
            cycle_after_last_operation_at: vec![],
            cycle: 0,
            current_operation_id,
            operations: vec![],
            arbitrary_angle_rotation_map: vec![],
        }
    }

    pub fn width(&self) -> u32 {
        self.conf.width
    }

    pub fn height(&self) -> u32 {
        self.conf.height
    }

    pub fn configuration(&self) -> &Configuration {
        &self.conf
    }

    fn issue_operation_id(&mut self) -> OperationId {
        let id = self.current_operation_id;
        self.current_operation_id.increment();
        id
    }

    pub fn operations(&self) -> &[OperationWithAdditionalData] {
        &self.operations
    }

    pub fn schedule(&mut self, op: &Operation) -> bool {
        match op {
            Operation::Measurement(axis) => {
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
            Operation::PauliRotation(rotation) => self.schedule_rotation(rotation),
        }
    }

    pub fn increment_cycle(&mut self) {
        self.cycle += 1;
    }

    pub fn set_cycle(&mut self, cycle: u32) {
        self.cycle = cycle;
    }

    pub fn cycle(&self) -> u32 {
        self.cycle
    }

    pub fn set_arbitrary_angle_rotation_map(&mut self, map: Vec<(f64, Vec<Pauli>, Vec<Pauli>)>) {
        self.arbitrary_angle_rotation_map = map;
    }

    fn schedule_mesuarement(&mut self, qubit: Qubit, axis: Pauli) -> bool {
        let (x, y) = self.data_qubit_mapping[&qubit];

        let duration = match axis {
            Pauli::X => 1,
            Pauli::Y => (self.conf.code_distance + 1) / 2,
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
        // TODO: We should push the measurement operation to `self.operations`.
        self.set_cycle_after_last_operation_at(qubit, self.cycle + self.cycle + duration);

        true
    }

    fn schedule_rotation(&mut self, rotation: &PauliRotation) -> bool {
        match rotation.angle {
            Angle::Zero => true,
            Angle::PiOver2 => true,
            Angle::PiOver4 => self.schedule_pi_over_4_rotation(rotation),
            Angle::PiOver8 => self.schedule_pi_over_8_rotation(rotation),
            Angle::Arbitrary(angle) => {
                let eps = self.conf.single_qubit_arbitrary_angle_rotation_precision;
                if let Some((_, pi_over_8_rotation_axes, pi_over_4_rotation_axes)) = self
                    .arbitrary_angle_rotation_map
                    .iter()
                    .find(|(a, _, _)| (*a - angle).abs() < eps)
                {
                    let support_size = rotation.axis.iter().filter(|a| **a != Pauli::I).count();
                    assert_eq!(support_size, 1);
                    let target_position =
                        rotation.axis.iter().position(|a| *a != Pauli::I).unwrap();
                    let target = Qubit::new(target_position);
                    let pi_over_8_rotation_axes = pi_over_8_rotation_axes.clone();
                    let pi_over_4_rotation_axes = pi_over_4_rotation_axes.clone();
                    self.schedule_single_qubit_pi_over_8_rotation_block(
                        target,
                        &pi_over_8_rotation_axes,
                        &pi_over_4_rotation_axes,
                    )
                } else {
                    panic!("self.arbitrary_angle_rotation_map.get(&{}) is None", angle);
                }
            }
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
        let distance = self.conf.code_distance;

        self.ensure_board_occupancy(cycle + distance + y_initialization_cost(distance));
        match axis {
            Pauli::I => true,
            Pauli::X => {
                let mut candidates = vec![];
                if x > 0 {
                    candidates.push((x - 1, y));
                }
                if x < self.conf.width - 1 {
                    candidates.push((x + 1, y));
                }
                self.schedule_pi_over_4_rotation_internal(
                    &[(qubit, x, y, axis)],
                    &[(x, y)],
                    &candidates,
                )
            }
            Pauli::Z => {
                let mut candidates = vec![];
                if y > 0 {
                    candidates.push((x, y - 1));
                }
                if y < self.conf.height - 1 {
                    candidates.push((x, y + 1));
                }
                self.schedule_pi_over_4_rotation_internal(
                    &[(qubit, x, y, axis)],
                    &[(x, y)],
                    &candidates,
                )
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
        let distance = self.conf.code_distance;
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
            self.add_neighbors(x, y, &mut candidates);
            adjacent_ancilla_candidates.extend(
                candidates
                    .iter()
                    .filter(|&(cx, cy)| !path.contains(&(*cx, *cy))),
            );
        }

        self.schedule_pi_over_4_rotation_internal(
            &[(q1, x1, y1, axis1), (q2, x2, y2, axis2)],
            &path,
            &adjacent_ancilla_candidates,
        )
    }

    fn schedule_pi_over_4_rotation_internal(
        &mut self,
        targets: &[(Qubit, u32, u32, Pauli)],
        path: &[(u32, u32)],
        ancilla_candidates: &[(u32, u32)],
    ) -> bool {
        use BoardOccupancy::*;
        let cycle = self.cycle;
        let distance = self.conf.code_distance;
        let y_initialization_cost = y_initialization_cost(distance);
        let id = self.issue_operation_id();

        let mut found = false;
        let mut ancilla_qubits = vec![];
        let conf = &self.conf;

        if conf.enable_two_qubit_pi_over_4_rotation_with_y_initialization
            && cycle >= y_initialization_cost
        {
            for &(x, y) in ancilla_candidates {
                if self.is_vacant(x, y, cycle - y_initialization_cost..cycle + distance) {
                    let cycle_range = cycle - y_initialization_cost..cycle;
                    self.set_occupancy_range(x, y, cycle_range, YInitialization(id));
                    let cycle_range = cycle..cycle + distance;
                    self.set_occupancy_range(x, y, cycle_range, LatticeSurgery(id));
                    for (q, _, _, _) in targets {
                        self.set_cycle_after_last_operation_at(*q, cycle + distance);
                    }
                    ancilla_qubits.push(Position { x, y });
                    found = true;
                    break;
                }
            }
        }

        if !found {
            for &(x, y) in ancilla_candidates {
                if self.is_vacant(x, y, cycle..cycle + distance + y_initialization_cost) {
                    let cycle_range = cycle..cycle + distance;
                    self.set_occupancy_range(x, y, cycle_range, LatticeSurgery(id));
                    let cycle_range =
                        cycle + distance..cycle + distance + y_measurement_cost(distance);
                    self.set_occupancy_range(x, y, cycle_range, YMeasurement(id));
                    for (q, _, _, _) in targets {
                        self.set_cycle_after_last_operation_at(*q, cycle + distance);
                    }
                    ancilla_qubits.push(Position { x, y });
                    found = true;
                    break;
                }
            }
        }

        if !found {
            return false;
        }

        for &(x, y) in path {
            let in_targets = targets.iter().any(|&(_, xt, yt, _)| xt == x && yt == y);
            let occupancy = if in_targets {
                BoardOccupancy::DataQubitInOperation(id)
            } else {
                BoardOccupancy::LatticeSurgery(id)
            };
            self.set_occupancy_range(x, y, cycle..cycle + distance, occupancy);
            if !in_targets {
                ancilla_qubits.push(Position { x, y });
            }
        }
        self.operations
            .push(OperationWithAdditionalData::PiOver4Rotation {
                id,
                targets: targets
                    .iter()
                    .map(|&(_, x, y, axis)| (Position { x, y }, axis))
                    .collect(),
                ancilla_qubits,
            });
        true
    }

    fn neighbors_with_xz_axis(
        x: u32,
        y: u32,
        axis: Pauli,
        available: &AncillaAvailability,
    ) -> Vec<(u32, u32)> {
        match axis {
            Pauli::I | Pauli::Y => {
                unreachable!();
            }
            Pauli::X => {
                let mut points = vec![];
                if x > 0 && available[(x - 1, y)] {
                    points.push((x - 1, y));
                }
                if x + 1 < available.width() && available[(x + 1, y)] {
                    points.push((x + 1, y));
                }
                points
            }
            Pauli::Z => {
                let mut points = vec![];
                if y > 0 && available[(x, y - 1)] {
                    points.push((x, y - 1));
                }
                if y + 1 < available.height() && available[(x, y + 1)] {
                    points.push((x, y + 1));
                }
                points
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn neighbors_with_y_axis(
        x: u32,
        y: u32,
        available: &AncillaAvailability,
    ) -> Vec<((u32, u32), (u32, u32), (u32, u32))> {
        let mut result = vec![];
        let width = available.width();
        let height = available.height();
        let mut push = |p1, p2, p3| {
            if available[p1] && available[p2] && available[p3] {
                result.push((p1, p2, p3));
            }
        };
        if x > 0 && y > 0 {
            push((x - 1, y), (x - 1, y - 1), (x, y - 1));
        }
        if x > 0 && y + 1 < height {
            push((x - 1, y), (x - 1, y + 1), (x, y + 1));
        }
        if x + 1 < width && y > 0 {
            push((x + 1, y), (x + 1, y - 1), (x, y - 1));
        }
        if x + 1 < width && y + 1 < height {
            push((x + 1, y), (x + 1, y + 1), (x, y + 1));
        }
        result
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
            Pauli::X | Pauli::Z => {
                for (x, y) in Self::neighbors_with_xz_axis(x1, y1, p1, available) {
                    q.push_back(((x, y), (x1, y1)));
                }
            }
            Pauli::Y => {
                for (p1, p2, p3) in Self::neighbors_with_y_axis(x1, y1, available) {
                    q.push_back((p1, (x1, y1)));
                    q.push_back((p2, (x1, y1)));
                    q.push_back((p3, (x1, y1)));
                    src_y_history.push([p1, p2, p3]);
                }
            }
        }

        let mut destinations = vec![];
        match p2 {
            Pauli::I => {
                unreachable!("path_between: Pauli::I");
            }
            Pauli::X | Pauli::Z => {
                for (x, y) in Self::neighbors_with_xz_axis(x2, y2, p2, available) {
                    destinations.push((x, y));
                }
            }
            Pauli::Y => {
                for (p1, p2, p3) in Self::neighbors_with_y_axis(x2, y2, available) {
                    destinations.push(p1);
                    destinations.push(p2);
                    destinations.push(p3);
                    dest_y_history.push([p1, p2, p3]);
                }
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
                            let filtered_region = region
                                .iter()
                                .filter(|&p| !path.contains(p))
                                .collect::<Vec<_>>();
                            path.extend(filtered_region);
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
            let target_position = rotation.axis.iter().position(|a| *a != Pauli::I).unwrap();
            let target = Qubit::new(target_position);
            self.schedule_single_qubit_pi_over_8_rotation(
                target,
                rotation.axis[target_position],
                self.conf.num_distillations_for_pi_over_8_rotation,
                self.conf.preferable_distillation_area_size,
            )
        } else {
            unimplemented!("schedule_pi_over_8_rotation: support size > 1");
        }
    }

    fn get_num_preflight_distillations(&self, x: u32, y: u32, cycle: u32) -> u32 {
        let mut c = cycle;
        while c > 0 && self.get_occupancy(x, y, c) == BoardOccupancy::Vacant {
            c -= 1;
        }
        (cycle - c) / self.conf.magic_state_distillation_cost
    }

    fn add_neighbors(&self, x: u32, y: u32, points: &mut Vec<(u32, u32)>) {
        let width = self.conf.width;
        let height = self.conf.height;
        if x > 0 {
            points.push((x - 1, y));
        }
        if x + 1 < width {
            points.push((x + 1, y));
        }
        if y > 0 {
            points.push((x, y - 1));
        }
        if y + 1 < height {
            points.push((x, y + 1));
        }
    }

    fn l1_distance((x1, y1): (u32, u32), (x2, y2): (u32, u32)) -> u32 {
        x1.abs_diff(x2) + y1.abs_diff(y2)
    }

    fn schedule_single_qubit_pi_over_8_rotation(
        &mut self,
        qubit: Qubit,
        axis: Pauli,
        num_distillations: u32,
        preferable_distillation_area_size: u32,
    ) -> bool {
        assert!(preferable_distillation_area_size > 0);
        let (x, y) = self.data_qubit_mapping[&qubit];
        let width = self.conf.width;
        let height = self.conf.height;
        let distance = self.conf.code_distance;
        let cycle = self.cycle;
        let up_to = cycle + distance + y_measurement_cost(distance);
        self.ensure_board_occupancy(up_to);
        if self.has_schedule_at_or_after(qubit, cycle) {
            return false;
        }

        let mut q = VecDeque::new();
        match axis {
            Pauli::I => return true,
            Pauli::X => {
                if x > 0 {
                    q.push_back((0_u32, vec![(x - 1, y)], vec![]));
                }
                if x + 1 < width {
                    q.push_back((0_u32, vec![(x + 1, y)], vec![]));
                }
            }
            Pauli::Z => {
                if y > 0 {
                    q.push_back((0_u32, vec![(x, y - 1)], vec![]));
                }
                if y + 1 < height {
                    q.push_back((0_u32, vec![(x, y + 1)], vec![]));
                }
            }
            Pauli::Y => {
                let mut routing_patterns = vec![];
                // We use this to avoid integer underflow. In any case, a value with underflow will
                // not be used because it will be filtered out by `cond1` and `cond2`.
                let pre = |x: u32| x.wrapping_sub(1);
                let available = |(x, y)| self.is_vacant(x, y, cycle..cycle + distance);
                let mut run = |cond1, cond2, p1, p2, p3| {
                    if cond1 && cond2 && available(p1) && available(p2) && available(p3) {
                        routing_patterns.push((p1, p2, p3));
                    }
                };

                run(x > 0, y > 0, (pre(x), y), (pre(x), pre(y)), (x, pre(y)));
                run(x > 0, y + 1 < height, (pre(x), y), (pre(x), y + 1), (x, y + 1));
                run(x + 1 < width, y > 0, (x + 1, y), (x + 1, pre(y)), (x, pre(y)));
                run(x + 1 < width, y + 1 < height, (x + 1, y), (x + 1, y + 1), (x, y + 1));

                for ((x1, y1), (x2, y2), (x3, y3)) in routing_patterns {
                    let mut points = vec![];
                    self.add_neighbors(x1, y1, &mut points);
                    self.add_neighbors(x2, y2, &mut points);
                    self.add_neighbors(x3, y3, &mut points);

                    for p in points {
                        if p != (x, y) && p != (x1, y1) && p != (x2, y2) && p != (x3, y3) {
                            q.push_back((0_u32, vec![p], vec![(x1, y1), (x2, y2), (x3, y3)]));
                        }
                    }
                }
            }
        }

        let mut visited = HashSet::new();
        let mut candidates = vec![];

        while let Some((num_distillations_so_far, history, routing_qubits)) = q.pop_front() {
            assert!(!history.is_empty());
            let mut sorted_history = history.clone();
            sorted_history.sort();
            if visited.contains(&sorted_history) {
                continue;
            }
            visited.insert(sorted_history);

            let limit = self.conf.num_distillations_for_pi_over_8_rotation;
            if Self::l1_distance(history[0], history[history.len() - 1]) > limit {
                continue;
            }

            let (x, y) = history[history.len() - 1];
            if !self.is_vacant(x, y, cycle..cycle + distance) {
                // This position is not eligible.
                continue;
            }

            let num_distillations_so_far =
                num_distillations_so_far + self.get_num_preflight_distillations(x, y, cycle);

            if num_distillations_so_far >= num_distillations {
                candidates.push((
                    num_distillations_so_far,
                    history.clone(),
                    routing_qubits.clone(),
                ));
            }

            if history.len() >= preferable_distillation_area_size as usize {
                continue;
            }

            if x > 0 && !history.contains(&(x - 1, y)) && !routing_qubits.contains(&(x - 1, y)) {
                let mut new_history = history.clone();
                new_history.push((x - 1, y));
                q.push_back((num_distillations_so_far, new_history, routing_qubits.clone()));
            }
            if x + 1 < width
                && !history.contains(&(x + 1, y))
                && !routing_qubits.contains(&(x + 1, y))
            {
                let mut new_history = history.clone();
                new_history.push((x + 1, y));
                q.push_back((num_distillations_so_far, new_history, routing_qubits.clone()));
            }
            if y > 0 && !history.contains(&(x, y - 1)) && !routing_qubits.contains(&(x, y - 1)) {
                let mut new_history = history.clone();
                new_history.push((x, y - 1));
                q.push_back((num_distillations_so_far, new_history, routing_qubits.clone()));
            }
            if y + 1 < height
                && !history.contains(&(x, y + 1))
                && !routing_qubits.contains(&(x, y + 1))
            {
                let mut new_history = history;
                new_history.push((x, y + 1));
                q.push_back((num_distillations_so_far, new_history, routing_qubits));
            }
        }

        if candidates.is_empty() {
            return false;
        }

        candidates.sort_by(|(ad, ahistory, _), (bd, bhistory, _)| {
            ahistory.len().cmp(&bhistory.len()).then_with(|| bd.cmp(ad))
        });

        for (_, distillation_area, routing_qubits) in &candidates {
            if self.place_gates_for_single_qubit_pi_over_8_rotation(
                x,
                y,
                axis,
                distillation_area,
                routing_qubits,
            ) {
                self.set_cycle_after_last_operation_at(qubit, cycle + distance);
                return true;
            }
        }

        false
    }

    fn place_gates_for_single_qubit_pi_over_8_rotation(
        &mut self,
        x: u32,
        y: u32,
        axis: Pauli,
        distillation_area: &[(u32, u32)],
        routing_qubits: &[(u32, u32)],
    ) -> bool {
        let cycle = self.cycle;
        let distance = self.conf.code_distance;
        let distillation_cost = self.conf.magic_state_distillation_cost;
        let y_measurement_cost = y_measurement_cost(distance);
        let correction_range = cycle + distance..cycle + distance + y_measurement_cost;

        for &(cx, cy) in distillation_area {
            if !self.is_vacant(cx, cy, correction_range.clone()) {
                return false;
            }
        }

        let id = self.issue_operation_id();

        for c in cycle..cycle + distance {
            self.set_occupancy(x, y, c, BoardOccupancy::DataQubitInOperation(id));
        }
        for &(x, y) in routing_qubits {
            for c in cycle..cycle + distance {
                self.set_occupancy(x, y, c, BoardOccupancy::LatticeSurgery(id));
            }
        }
        let mut num_distillations = 0_u32;
        for &(x, y) in distillation_area {
            let mut c = cycle;
            while c >= distillation_cost && self.is_vacant(x, y, c - distillation_cost..c) {
                c -= distillation_cost;
                num_distillations += 1;
            }
        }
        for &(x, y) in distillation_area {
            let occupancy = BoardOccupancy::MagicStateDistillation(id);
            let mut c = cycle;
            let limit = self.conf.num_distillations_for_pi_over_8_rotation;
            while c >= distillation_cost && self.is_vacant(x, y, c - distillation_cost..c) {
                c -= distillation_cost;
                if (cycle - c) > distillation_cost * limit {
                    break;
                }
            }
            let distillation_start = c;
            self.set_occupancy_range(x, y, distillation_start..cycle, occupancy);
            self.set_occupancy_range(
                x,
                y,
                cycle..cycle + distance,
                BoardOccupancy::LatticeSurgery(id),
            );
        }

        for &(cx, cy) in distillation_area {
            let occupancy = BoardOccupancy::YMeasurement(id);
            self.set_occupancy_range(cx, cy, correction_range.clone(), occupancy);
        }
        self.operations
            .push(OperationWithAdditionalData::PiOver8Rotation {
                id,
                targets: vec![(Position { x, y }, axis)],
                routing_qubits: routing_qubits
                    .iter()
                    .map(|&(x, y)| Position { x, y })
                    .collect(),
                distillation_qubits: distillation_area
                    .iter()
                    .map(|&(x, y)| Position { x, y })
                    .collect(),
                num_distillations,
                num_distillations_on_retry: distillation_area.len() as u32,
            });

        true
    }

    fn schedule_single_qubit_pi_over_8_rotation_block(
        &mut self,
        qubit: Qubit,
        pi_over_8_rotation_axes: &[Pauli],
        pi_over_4_rotation_axes: &[Pauli],
    ) -> bool {
        let distillation_area_size = self.conf.num_distillations_for_pi_over_8_rotation_block;
        let width = self.conf.width;
        let height = self.conf.height;
        assert!(distillation_area_size > 1);
        let cycle = self.cycle;
        let distance = self.conf.code_distance;
        let expected_time_cost = self.expected_time_cost_for_single_qubit_pi_over_8_rotation_block(
            pi_over_8_rotation_axes.len() as u32,
            distillation_area_size,
        );
        self.ensure_board_occupancy(cycle + expected_time_cost + y_measurement_cost(distance));
        if self.has_schedule_at_or_after(qubit, cycle) {
            return false;
        }
        let block_cycle_range = cycle..cycle + expected_time_cost;
        let (x, y) = self.data_qubit_mapping[&qubit];
        let mut routing_qubits_candidates = vec![];
        if x > 0 && y > 0 {
            routing_qubits_candidates.push([(x - 1, y), (x, y - 1), (x - 1, y - 1)]);
        }
        if x > 0 && y + 1 < height {
            routing_qubits_candidates.push([(x - 1, y), (x, y + 1), (x - 1, y + 1)]);
        }
        if x + 1 < width && y > 0 {
            routing_qubits_candidates.push([(x + 1, y), (x, y - 1), (x + 1, y - 1)]);
        }
        if x + 1 < width && y + 1 < height {
            routing_qubits_candidates.push([(x + 1, y), (x, y + 1), (x + 1, y + 1)]);
        }

        let mut q = routing_qubits_candidates
            .iter()
            .filter_map(|candidate| {
                let is_vacant = |&(x, y)| self.is_vacant(x, y, block_cycle_range.clone());
                if candidate.iter().all(is_vacant) {
                    Some((candidate, Vec::<(u32, u32)>::new()))
                } else {
                    None
                }
            })
            .collect::<VecDeque<_>>();

        let mut result_candidates = Vec::new();
        let mut visited = HashSet::new();
        while let Some((routing_qubits, distillation_qubits)) = q.pop_front() {
            if distillation_qubits.len() == distillation_area_size as usize {
                let num_distillation_qubits_adjacent_to_routing_qubits = distillation_qubits
                    .iter()
                    .filter(|&&(x, y)| {
                        let is_adjacent =
                            |&(rx, ry): &(u32, u32)| rx.abs_diff(x) + ry.abs_diff(y) == 1;
                        routing_qubits.iter().any(is_adjacent)
                    })
                    .count()
                    as u32;

                let mut q = VecDeque::new();
                let mut visited = HashSet::new();
                let mut max_distance = 0_u32;
                for &(x, y) in routing_qubits {
                    q.push_back((x, y, 0));
                }
                while let Some((x, y, d)) = q.pop_front() {
                    if visited.contains(&(x, y)) {
                        continue;
                    }
                    visited.insert((x, y));
                    max_distance = std::cmp::max(max_distance, d);
                    let mut push = |(x, y)| {
                        if x >= width || y >= height {
                            return;
                        }
                        if !distillation_qubits.contains(&(x, y)) {
                            return;
                        }
                        q.push_back((x, y, d + 1));
                    };

                    if x > 0 {
                        push((x - 1, y));
                    }
                    if y > 0 {
                        push((x, y - 1));
                    }
                    push((x + 1, y));
                    push((x, y + 1));
                }

                let entry = (
                    routing_qubits,
                    distillation_qubits,
                    num_distillation_qubits_adjacent_to_routing_qubits,
                    max_distance,
                );

                result_candidates.push(entry);
                continue;
            }

            let mut next_candidates = vec![];
            let mut push = |x, y| {
                if x >= width || y >= height {
                    return;
                }
                let mut existing = next_candidates
                    .iter()
                    .chain(routing_qubits.iter())
                    .chain(distillation_qubits.iter());
                if existing.any(|&(cx, cy)| cx == x && cy == y) {
                    return;
                }
                if !self.is_vacant(x, y, block_cycle_range.clone()) {
                    return;
                }
                next_candidates.push((x, y));
            };

            routing_qubits
                .iter()
                .chain(distillation_qubits.iter())
                .for_each(|&(x, y)| {
                    if x > 0 {
                        push(x - 1, y);
                    }
                    if y > 0 {
                        push(x, y - 1);
                    }
                    push(x + 1, y);
                    push(x, y + 1);
                });
            for next in &next_candidates {
                let mut new_distillation_qubits = distillation_qubits.clone();
                new_distillation_qubits.push(*next);
                // Sort the qubits to check the uniqueness.
                new_distillation_qubits.sort();

                let entry = (routing_qubits, new_distillation_qubits);
                if visited.contains(&entry) {
                    continue;
                }
                visited.insert(entry.clone());
                q.push_back(entry);
            }
        }

        result_candidates.sort_by(
            |(_, _, num_adjacent1, max_distance1), &(_, _, num_adjacent2, max_distance2)| {
                num_adjacent1
                    .cmp(&num_adjacent2)
                    .reverse()
                    .then(max_distance1.cmp(&max_distance2))
            },
        );

        for (routing_qubits, distillation_qubits, _, _) in result_candidates {
            if self.place_gates_for_single_qubit_pi_over_8_rotation_block(
                x,
                y,
                pi_over_8_rotation_axes,
                pi_over_4_rotation_axes,
                routing_qubits,
                &distillation_qubits,
            ) {
                self.set_cycle_after_last_operation_at(qubit, cycle + expected_time_cost);
                return true;
            }
        }
        false
    }

    fn expected_time_cost_for_single_qubit_pi_over_8_rotation_block(
        &self,
        depth: u32,
        num_distillation_blocks: u32,
    ) -> u32 {
        let distance = self.conf.code_distance;
        let distillation_cost = self.conf.magic_state_distillation_cost;
        let success_rate = self.conf.magic_state_distillation_success_rate;
        let expected_distillation_cost =
            distillation_cost as f64 / success_rate / num_distillation_blocks as f64;
        let ratio = self.conf.single_qubit_pi_over_8_rotation_block_depth_ratio;

        let first_round_cost =
            std::cmp::max((expected_distillation_cost * ratio).ceil() as u32, distillation_cost)
                + distance;
        let one_round_cost =
            std::cmp::max((expected_distillation_cost * ratio).ceil() as u32, distance);

        first_round_cost + one_round_cost * (depth - 1) + 2 * distance
    }

    fn translate<T: std::ops::Add<Output = T> + Copy>(r: Range<T>, x: T) -> Range<T> {
        r.start + x..r.end + x
    }

    fn place_gates_for_single_qubit_pi_over_8_rotation_block(
        &mut self,
        x: u32,
        y: u32,
        pi_over_8_rotation_axes: &[Pauli],
        pi_over_4_rotation_axes: &[Pauli],
        routing_qubits: &[(u32, u32)],
        distillation_qubits: &[(u32, u32)],
    ) -> bool {
        use BoardOccupancy::*;
        let cycle = self.cycle;
        let distance = self.conf.code_distance;
        let expected_time_cost = self.expected_time_cost_for_single_qubit_pi_over_8_rotation_block(
            pi_over_8_rotation_axes.len() as u32,
            distillation_qubits.len() as u32,
        );
        // 2 * distance for the last two Clifford corrections (excluding Y measurements).
        assert!(expected_time_cost > 2 * distance);
        let block_time_cost: u32 = expected_time_cost - 2 * distance;
        let block_cycle_range = cycle..cycle + block_time_cost;
        let y_measurement_cost = y_measurement_cost(distance);
        let second_last_correction_range = Self::translate(0..distance, cycle + block_time_cost);
        let second_last_y_measurement_range =
            Self::translate(0..y_measurement_cost, second_last_correction_range.end);
        let last_correction_range =
            Self::translate(0..distance, cycle + block_time_cost + distance);
        let last_y_measurement_range =
            Self::translate(0..y_measurement_cost, last_correction_range.end);

        assert!(!distillation_qubits.is_empty());
        assert_eq!(routing_qubits.len(), 3);
        // We choose one qubit for the second-last Clifford correction from `distillation_qubits`,
        // and one qubit for the last Clifford correction from `routing_qubits`. This implies that
        // the last Clifford correction's axis cannot be Y. We can guarantee that at runtime.
        // Although we can choose other qubits for correction qubits, we choose these qubits for
        // simplicity.
        let (cx1, cy1) = distillation_qubits[0];
        let (cx2, cy2) = routing_qubits[0];
        let (cx3, cy3) = routing_qubits[1];

        assert_eq!(cy2, y);
        assert_eq!(cx3, x);

        if !self.is_vacant(cx1, cy1, second_last_correction_range.clone()) {
            return false;
        }
        if !self.is_vacant(cx2, cy2, last_correction_range.clone()) {
            return false;
        }
        if !self.is_vacant(cx2, cy2, last_y_measurement_range.clone()) {
            return false;
        }
        if !self.is_vacant(cx3, cy3, last_y_measurement_range.clone()) {
            return false;
        }

        let id = self.issue_operation_id();
        self.set_occupancy_range(x, y, cycle..last_correction_range.end, DataQubitInOperation(id));

        let occupancy = PiOver8RotationBlock(id);
        for &(x, y) in routing_qubits.iter().chain(distillation_qubits) {
            self.set_occupancy_range(x, y, block_cycle_range.clone(), occupancy.clone());
        }

        // The lattice surgery operation for the second-last Clifford correction.
        for &(x, y) in routing_qubits.iter().chain(distillation_qubits) {
            self.set_occupancy_range(x, y, second_last_correction_range.clone(), occupancy.clone());
        }

        // The Y measurement for the second-last Clifford correction.
        self.set_occupancy_range(cx1, cy1, second_last_y_measurement_range, occupancy.clone());

        // The lattice surgery operation for the last Clifford correction.
        for &(x, y) in routing_qubits {
            self.set_occupancy_range(x, y, last_correction_range.clone(), occupancy.clone());
        }

        // The Y measurement for the last Clifford correction.
        self.set_occupancy_range(cx2, cy2, last_y_measurement_range.clone(), occupancy.clone());
        self.set_occupancy_range(cx3, cy3, last_y_measurement_range, occupancy.clone());

        let correction_qubits = vec![
            Position { x: cx1, y: cy1 },
            Position { x: cx2, y: cy2 },
            Position { x: cx3, y: cy3 },
        ];
        self.operations
            .push(OperationWithAdditionalData::SingleQubitPiOver8RotationBlock {
                id,
                target: Position { x, y },
                routing_qubits: routing_qubits
                    .iter()
                    .map(|&(x, y)| Position { x, y })
                    .collect(),
                distillation_qubits: distillation_qubits
                    .iter()
                    .map(|&(x, y)| Position { x, y })
                    .collect(),
                correction_qubits,
                pi_over_8_axes: pi_over_8_rotation_axes.to_vec(),
                pi_over_4_axes: pi_over_4_rotation_axes.to_vec(),
            });

        true
    }

    fn ensure_board_occupancy(&mut self, cycle: u32) {
        let size = (self.conf.width * self.conf.height) as usize;
        assert_eq!(self.occupancy.len() % size, 0);
        let current_max_cycle = (self.occupancy.len() / size) as u32;

        if cycle < current_max_cycle {
            // There is nothing to do.
            return;
        }

        let new_size = ((cycle + 1) as usize) * size;
        self.occupancy.resize(new_size, BoardOccupancy::Vacant);
        for c in current_max_cycle..cycle + 1 {
            for (x, y) in self.data_qubit_mapping.values() {
                let o = BoardOccupancy::IdleDataQubit;
                // We don't use set_occupancy here in order to avoid Rust borrow checker complaints.
                let index =
                    (c * self.conf.width * self.conf.height + y * self.conf.width + x) as usize;
                self.occupancy[index] = o;
            }
        }
    }

    // Returns the end cycle of the last operation.
    pub fn get_last_end_cycle(&self) -> u32 {
        let mut cycle = self
            .cycle_after_last_operation_at
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let size = (self.conf.width * self.conf.height) as usize;
        let mut last_end_cycle = cycle;

        while cycle < (self.occupancy.len() / size) as u32 {
            for x in 0..self.conf.width {
                for y in 0..self.conf.height {
                    if !self.get_occupancy(x, y, cycle).is_vacant_or_idle() {
                        last_end_cycle = cycle + 1;
                    }
                }
            }

            cycle += 1;
        }
        last_end_cycle
    }

    pub fn get_occupancy(&self, x: u32, y: u32, cycle: u32) -> BoardOccupancy {
        assert!(x < self.conf.width);
        assert!(y < self.conf.height);
        let size = (self.conf.width * self.conf.height) as usize;
        assert_eq!(self.occupancy.len() % size, 0);
        if cycle >= (self.occupancy.len() / size) as u32 {
            if self
                .data_qubit_mapping
                .values()
                .any(|(cx, cy)| *cx == x && *cy == y)
            {
                return BoardOccupancy::IdleDataQubit;
            } else {
                return BoardOccupancy::Vacant;
            }
        }

        let index = (cycle * self.conf.width * self.conf.height + y * self.conf.width + x) as usize;
        self.occupancy[index].clone()
    }

    // Returns true if any operation is scheduled for `qubit` at or after `cycle`.
    fn has_schedule_at_or_after(&self, qubit: Qubit, cycle: u32) -> bool {
        if qubit.qubit >= self.cycle_after_last_operation_at.len() {
            return false;
        }
        self.cycle_after_last_operation_at[qubit.qubit] > cycle
    }

    pub fn get_earliest_available_cycle_at(&self, qubit: Qubit) -> u32 {
        if qubit.qubit >= self.cycle_after_last_operation_at.len() {
            return 0;
        }
        self.cycle_after_last_operation_at[qubit.qubit]
    }

    // Returns true if the qubit at (x, y) is `occupancy` at all the cycles in the range.
    pub fn is_occupancy(
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
        assert!(x < self.conf.width);
        assert!(y < self.conf.height);
        let size = (self.conf.width * self.conf.height) as usize;
        assert_eq!(self.occupancy.len() % size, 0);
        assert!(cycle < (self.occupancy.len() / size) as u32);
        let index = (cycle * self.conf.width * self.conf.height + y * self.conf.width + x) as usize;
        assert!(self.occupancy[index].is_vacant_or_idle());

        self.occupancy[index] = occupancy;
    }

    fn set_occupancy_range(
        &mut self,
        x: u32,
        y: u32,
        cycle_range: Range<u32>,
        occupancy: BoardOccupancy,
    ) {
        for c in cycle_range {
            self.set_occupancy(x, y, c, occupancy.clone());
        }
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

impl<T: Clone> Map2D<T> {
    pub fn new_with_value(width: u32, height: u32, value: T) -> Self {
        Map2D {
            width,
            height,
            map: vec![value; (width * height) as usize],
        }
    }
}

impl<T: Clone> Index<(u32, u32)> for Map2D<T> {
    type Output = T;

    fn index(&self, index: (u32, u32)) -> &Self::Output {
        let (x, y) = index;
        assert!(x < self.width);
        assert!(y < self.height);
        &self.map[(y * self.width + x) as usize]
    }
}

impl<T: Clone> IndexMut<(u32, u32)> for Map2D<T> {
    fn index_mut(&mut self, index: (u32, u32)) -> &mut Self::Output {
        let (x, y) = index;
        assert!(x < self.width);
        assert!(y < self.height);
        &mut self.map[(y * self.width + x) as usize]
    }
}

impl AncillaAvailability {
    fn new(board: &Board, cycle_range: Range<u32>) -> Self {
        let size = (board.width() * board.height()) as usize;
        let mut map = AncillaAvailability {
            map: Map2D {
                width: board.width(),
                height: board.height(),
                map: vec![false; size],
            },
        };

        for y in 0..board.height() {
            for x in 0..board.width() {
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

    fn default_conf() -> Configuration {
        Configuration {
            width: 5,
            height: 5,
            code_distance: 5,
            magic_state_distillation_cost: 10,
            num_distillations_for_pi_over_8_rotation: 1,
            magic_state_distillation_success_rate: 0.5,
            num_distillations_for_pi_over_8_rotation_block: 1,
            single_qubit_pi_over_8_rotation_block_depth_ratio: 1.2,
            single_qubit_arbitrary_angle_rotation_precision: 1e-10,
            preferable_distillation_area_size: 5,
            enable_two_qubit_pi_over_4_rotation_with_y_initialization: false,
        }
    }

    fn new_board(mapping: DataQubitMapping, code_distance: u32) -> Board {
        let conf = Configuration {
            width: mapping.width,
            height: mapping.height,
            code_distance,
            ..default_conf()
        };

        Board::new(mapping, &conf)
    }

    fn p(x: u32, y: u32) -> Position {
        Position { x, y }
    }

    #[test]
    fn test_init_board() {
        let mut mapping = DataQubitMapping::new(3, 4);
        mapping.map(Qubit::new(6), 0, 0);
        mapping.map(Qubit::new(7), 2, 2);

        let board = new_board(mapping, 5);

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
        let mut board = new_board(mapping, 5);

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
    fn test_get_last_end_cycle() {
        let mut mapping = DataQubitMapping::new(3, 4);
        let q6 = Qubit::new(6);
        let q7 = Qubit::new(7);
        mapping.map(q6, 0, 0);
        mapping.map(q7, 2, 2);
        let mut board = new_board(mapping, 5);
        let id = board.issue_operation_id();

        assert_eq!(board.get_last_end_cycle(), 0);

        board.set_cycle_after_last_operation_at(q6, 4);
        board.set_cycle_after_last_operation_at(q7, 2);

        assert_eq!(board.get_last_end_cycle(), 4);

        board.ensure_board_occupancy(12);

        assert_eq!(board.get_last_end_cycle(), 4);

        board.set_occupancy(0, 1, 3, BoardOccupancy::LatticeSurgery(id));

        assert_eq!(board.get_last_end_cycle(), 4);

        board.set_occupancy(0, 2, 4, BoardOccupancy::LatticeSurgery(id));

        assert_eq!(board.get_last_end_cycle(), 5);

        board.set_occupancy(0, 2, 10, BoardOccupancy::LatticeSurgery(id));

        assert_eq!(board.get_last_end_cycle(), 11);
    }

    #[test]
    fn test_get_set_occupancy() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 4);
        mapping.map(Qubit::new(6), 0, 0);
        mapping.map(Qubit::new(7), 2, 2);
        let mut board = new_board(mapping, 5);

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
        let mut board = new_board(mapping, 5);
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
        let mut board = new_board(mapping, 5);
        board.ensure_board_occupancy(6);

        let id = board.issue_operation_id();
        board.set_occupancy(0, 0, 0, DataQubitInOperation(id));
        board.set_cycle_after_last_operation_at(Qubit::new(0), 1);

        let id = board.issue_operation_id();
        board.set_occupancy(1, 0, 1, LatticeSurgery(id));

        // Rejected because the qubit is already in operation at cycle 0.
        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XII")
        })));

        board.cycle = 1;
        // Rejected because the ancilla qubit at (1, 0) is already used.
        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
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
        assert!(board.is_occupancy(1, 0, 7..11, YMeasurement(id)));
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

        let operations = vec![OperationWithAdditionalData::PiOver4Rotation {
            id,
            targets: vec![(Position { x: 0, y: 0 }, Pauli::X)],
            ancilla_qubits: vec![(Position { x: 1, y: 0 })],
        }];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_single_qubit_clifford_rotation_with_z_axis() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 3);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 2, 2);
        let mut board = new_board(mapping, 3);
        board.conf.enable_two_qubit_pi_over_4_rotation_with_y_initialization = true;
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
        assert!(board.is_occupancy(2, 1, 0..6, YInitialization(id)));
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

        let operations = vec![OperationWithAdditionalData::PiOver4Rotation {
            id,
            targets: vec![(Position { x: 2, y: 2 }, Pauli::Z)],
            ancilla_qubits: vec![(Position { x: 2, y: 1 })],
        }];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_single_qubit_clifford_rotation_with_z_axis_competing_neighbors() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 3);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 2, 1);
        let mut board = new_board(mapping, 3);
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
        assert!(board.is_occupancy(0, 2, 3..6, YMeasurement(OperationId { id: 1 })));
        assert!(board.is_occupancy(0, 2, 6..10, Vacant));
        assert!(board.is_occupancy(1, 0, 0..3, LatticeSurgery(OperationId { id: 2 })));
        assert!(board.is_occupancy(1, 0, 3..6, YMeasurement(OperationId { id: 2 })));
        assert!(board.is_occupancy(1, 0, 6..10, Vacant));
        assert!(board.is_occupancy(1, 1, 0..3, DataQubitInOperation(OperationId { id: 2 })));
        assert!(board.is_occupancy(1, 1, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(1, 2, 0..1, LatticeSurgery(OperationId { id: 0 })));
        assert!(board.is_occupancy(1, 2, 1..10, Vacant));
        assert!(board.is_occupancy(2, 0, 0..3, LatticeSurgery(OperationId { id: 3 })));
        assert!(board.is_occupancy(2, 0, 3..6, YMeasurement(OperationId { id: 3 })));
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

        let operations = vec![
            OperationWithAdditionalData::PiOver4Rotation {
                id: OperationId { id: 1 },
                targets: vec![(Position { x: 0, y: 1 }, Pauli::Z)],
                ancilla_qubits: vec![(Position { x: 0, y: 2 })],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: OperationId { id: 2 },
                targets: vec![(Position { x: 1, y: 1 }, Pauli::Z)],
                ancilla_qubits: vec![(Position { x: 1, y: 0 })],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: OperationId { id: 3 },
                targets: vec![(Position { x: 2, y: 1 }, Pauli::Z)],
                ancilla_qubits: vec![(Position { x: 2, y: 0 })],
            },
        ];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_single_qubit_clifford_rotation_with_x_axis_competing_neighbors() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(3, 3);
        mapping.map(Qubit::new(0), 1, 0);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 1, 2);
        let mut board = new_board(mapping, 3);
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
        assert!(board.is_occupancy(0, 1, 3..6, YMeasurement(OperationId { id: 3 })));
        assert!(board.is_occupancy(0, 1, 6..10, Vacant));
        assert!(board.is_occupancy(0, 2, 0..3, LatticeSurgery(OperationId { id: 4 })));
        assert!(board.is_occupancy(0, 2, 3..6, YMeasurement(OperationId { id: 4 })));
        assert!(board.is_occupancy(0, 2, 6..10, Vacant));
        assert!(board.is_occupancy(1, 0, 0..3, DataQubitInOperation(OperationId { id: 2 })));
        assert!(board.is_occupancy(1, 0, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(1, 1, 0..3, DataQubitInOperation(OperationId { id: 3 })));
        assert!(board.is_occupancy(1, 1, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(1, 2, 0..3, DataQubitInOperation(OperationId { id: 4 })));
        assert!(board.is_occupancy(1, 2, 3..10, IdleDataQubit));
        assert!(board.is_occupancy(2, 0, 0..3, LatticeSurgery(OperationId { id: 2 })));
        assert!(board.is_occupancy(2, 0, 3..6, YMeasurement(OperationId { id: 2 })));
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

        let operations = vec![
            OperationWithAdditionalData::PiOver4Rotation {
                id: OperationId { id: 2 },
                targets: vec![(Position { x: 1, y: 0 }, Pauli::X)],
                ancilla_qubits: vec![(Position { x: 2, y: 0 })],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: OperationId { id: 3 },
                targets: vec![(Position { x: 1, y: 1 }, Pauli::X)],
                ancilla_qubits: vec![(Position { x: 0, y: 1 })],
            },
            OperationWithAdditionalData::PiOver4Rotation {
                id: OperationId { id: 4 },
                targets: vec![(Position { x: 1, y: 2 }, Pauli::X)],
                ancilla_qubits: vec![(Position { x: 0, y: 2 })],
            },
        ];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_two_qubit_clifford_rotation_conflicting() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(4, 3);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        mapping.map(Qubit::new(2), 3, 2);
        let mut board = new_board(mapping, 3);

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
        assert_eq!(board.operations, vec![]);
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
        assert_eq!(
            Board::path_between((3, 1, Y), (3, 3, Y), &map),
            Some(vec![(3, 1), (3, 2), (2, 1), (2, 2), (2, 3), (3, 3)])
        );
    }

    #[test]
    fn test_schedule_two_qubit_clifford_rotation_data_qubit_is_busy() {
        use BoardOccupancy::*;
        let mut mapping = DataQubitMapping::new(2, 2);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        let mut board = new_board(mapping, 3);
        board.ensure_board_occupancy(0);
        let id = board.issue_operation_id();
        board.set_occupancy(0, 0, 0, DataQubitInOperation(id));
        board.set_cycle_after_last_operation_at(Qubit::new(0), 1);

        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XX")
        })));
    }

    #[test]
    fn test_schedule_two_qubit_clifford_rotation_unreachable() {
        let mut mapping = DataQubitMapping::new(2, 2);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 1);
        let mut board = new_board(mapping, 3);
        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
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
        let mut board = new_board(mapping, 3);
        board.ensure_board_occupancy(3);
        let id = board.issue_operation_id();
        board.set_occupancy(1, 0, 2, LatticeSurgery(id));
        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
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
        let mut board = new_board(mapping, 3);
        board.ensure_board_occupancy(3);

        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
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
        let mut board = new_board(mapping, 3);
        board.ensure_board_occupancy(3);
        let id = board.issue_operation_id();
        board.set_occupancy(1, 0, 3, LatticeSurgery(id));

        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
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
        let mut board = new_board(mapping, 3);
        board.ensure_board_occupancy(3);
        let id = board.issue_operation_id();
        board.set_occupancy(0, 1, 3, LatticeSurgery(id));
        board.set_occupancy(1, 1, 3, LatticeSurgery(id));

        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("ZZ")
        })));
    }

    #[test]
    fn test_schedule_two_qubit_clifford_rotation_no_space_for_y_eigenstate_4() {
        let mut mapping = DataQubitMapping::new(2, 4);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 0, 2);
        let mut board = new_board(mapping, 3);

        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("YY")
        })));
    }

    #[test]
    fn test_schedule_xz_pi_over_4_rotation_with_y_initialization() {
        use BoardOccupancy::*;
        let width = 2_u32;
        let height = 4_u32;
        let dummy_id = OperationId { id: 9999 };
        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 3);
        let mut board = new_board(mapping, 3);
        board.conf.enable_two_qubit_pi_over_4_rotation_with_y_initialization = true;
        board.ensure_board_occupancy(11);
        board.set_occupancy(1, 1, 3, YMeasurement(dummy_id));
        board.set_occupancy(1, 2, 4, YMeasurement(dummy_id));
        board.set_occupancy(1, 1, 11, YInitialization(dummy_id));
        let id = OperationId { id: 0 };

        board.cycle = 8;
        assert!(board.schedule(&Operation::PauliRotation(PauliRotation {
            angle: Angle::PiOver4,
            axis: new_axis("XZ")
        })));

        assert!(board.is_occupancy(0, 0, 0..8, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 0..2, Vacant));
        assert!(board.is_occupancy(0, 1, 2..8, YInitialization(id)));
        assert!(board.is_occupancy(0, 2, 0..8, Vacant));
        assert!(board.is_occupancy(0, 3, 0..8, Vacant));
        assert!(board.is_occupancy(1, 0, 0..8, Vacant));
        assert!(board.is_occupancy(1, 1, 0..3, Vacant));
        assert!(board.is_occupancy(1, 1, 3..4, YMeasurement(dummy_id)));
        assert!(board.is_occupancy(1, 1, 4..8, Vacant));
        assert!(board.is_occupancy(1, 2, 0..4, Vacant));
        assert!(board.is_occupancy(1, 2, 4..5, YMeasurement(dummy_id)));
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
        assert!(board.is_occupancy(1, 1, 11..12, YInitialization(dummy_id)));
        assert!(board.is_occupancy(1, 2, 11..12, Vacant));
        assert!(board.is_occupancy(1, 3, 11..12, IdleDataQubit));

        let operations = vec![OperationWithAdditionalData::PiOver4Rotation {
            id,
            targets: vec![
                (Position { x: 0, y: 0 }, Pauli::X),
                (Position { x: 1, y: 3 }, Pauli::Z),
            ],
            ancilla_qubits: vec![
                Position { x: 0, y: 1 },
                Position { x: 1, y: 0 },
                Position { x: 1, y: 1 },
                Position { x: 1, y: 2 },
            ],
        }];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_xz_pi_over_4_rotation_with_adjacent_y_measurement() {
        use BoardOccupancy::*;
        let width = 2_u32;
        let height = 4_u32;
        let dummy_id = OperationId { id: 9999 };
        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 0);
        mapping.map(Qubit::new(1), 1, 3);
        let mut board = new_board(mapping, 3);
        board.ensure_board_occupancy(4);
        board.set_occupancy(1, 1, 3, YInitialization(dummy_id));
        board.set_occupancy(1, 2, 4, YInitialization(dummy_id));
        let id = OperationId { id: 0 };

        assert!(board.schedule(&Operation::PauliRotation(PauliRotation {
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
        assert!(board.is_occupancy(0, 1, 3..6, YMeasurement(id)));
        assert!(board.is_occupancy(0, 2, 3..6, Vacant));
        assert!(board.is_occupancy(0, 3, 3..6, Vacant));
        assert!(board.is_occupancy(1, 0, 3..6, Vacant));
        assert!(board.is_occupancy(1, 1, 3..4, YInitialization(dummy_id)));
        assert!(board.is_occupancy(1, 1, 4..6, Vacant));
        assert!(board.is_occupancy(1, 2, 3..4, Vacant));
        assert!(board.is_occupancy(1, 2, 4..5, YInitialization(dummy_id)));
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

        let operations = vec![OperationWithAdditionalData::PiOver4Rotation {
            id,
            targets: vec![
                (Position { x: 0, y: 0 }, Pauli::X),
                (Position { x: 1, y: 3 }, Pauli::Z),
            ],
            ancilla_qubits: vec![
                Position { x: 0, y: 1 },
                Position { x: 1, y: 0 },
                Position { x: 1, y: 1 },
                Position { x: 1, y: 2 },
            ],
        }];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_yy_pi_over_4_rotation_with_adjacent_y_measurement() {
        use BoardOccupancy::*;
        let width = 3_u32;
        let height = 4_u32;
        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 0, 2);
        let mut board = new_board(mapping, 3);

        assert!(board.schedule(&Operation::PauliRotation(PauliRotation {
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
        assert!(board.is_occupancy(2, 0, 3..6, YMeasurement(id)));
        assert!(board.is_occupancy(2, 0, 6..7, Vacant));
        assert!(board.is_occupancy(2, 1, 3..7, Vacant));
        assert!(board.is_occupancy(2, 2, 3..7, Vacant));
        assert!(board.is_occupancy(2, 3, 3..7, Vacant));

        let operations = vec![OperationWithAdditionalData::PiOver4Rotation {
            id,
            targets: vec![
                (Position { x: 0, y: 1 }, Pauli::Y),
                (Position { x: 0, y: 2 }, Pauli::Y),
            ],
            ancilla_qubits: vec![
                Position { x: 2, y: 0 },
                Position { x: 0, y: 0 },
                Position { x: 1, y: 0 },
                Position { x: 1, y: 1 },
                Position { x: 0, y: 3 },
                Position { x: 1, y: 3 },
                Position { x: 1, y: 2 },
            ],
        }];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_pi_over_8_rotation_used_qubit() {
        let width = 3_u32;
        let height = 4_u32;
        let mut mapping = DataQubitMapping::new(width, height);
        let q = Qubit::new(0);
        mapping.map(q, 0, 1);
        mapping.map(Qubit::new(1), 0, 2);
        let conf = Configuration {
            width,
            height,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);

        board.set_cycle_after_last_operation_at(q, 1000);
        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
            angle: Angle::PiOver8,
            axis: new_axis("Z")
        })));

        board.cycle = 999;
        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
            angle: Angle::PiOver8,
            axis: new_axis("Z")
        })));
        assert_eq!(board.operations, vec![]);
    }

    #[test]
    fn test_schedule_pi_over_8_rotation_no_space_for_distillation() {
        use BoardOccupancy::*;
        let width = 3_u32;
        let height = 4_u32;
        let dummy_id = OperationId { id: 9999 };
        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 0, 2);
        let conf = Configuration {
            width,
            height,
            code_distance: 3,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);
        board.ensure_board_occupancy(10);
        board.set_occupancy(1, 1, 0, YMeasurement(dummy_id));
        board.set_occupancy(0, 1, 0, YMeasurement(dummy_id));
        board.set_occupancy(0, 3, 0, YInitialization(dummy_id));
        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
            angle: Angle::PiOver8,
            axis: new_axis("XI")
        })));

        assert!(!board.schedule(&Operation::PauliRotation(PauliRotation {
            angle: Angle::PiOver8,
            axis: new_axis("IZ")
        })));
        assert!(board.operations.is_empty());
    }

    #[test]
    fn test_schedule_pi_over_8_rotation_no_space_for_correction() {
        use BoardOccupancy::*;
        let width = 3_u32;
        let height = 4_u32;
        let dummy_id = OperationId { id: 9999 };
        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 0);
        let conf = Configuration {
            width,
            height,
            code_distance: 3,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);
        board.ensure_board_occupancy(10);
        board.set_occupancy(0, 1, 8, YMeasurement(dummy_id));

        board.cycle = 5;
        assert!(!board.schedule_single_qubit_pi_over_8_rotation(Qubit::new(0), Pauli::Z, 1, 1));
        assert!(board.operations.is_empty());
    }

    #[test]
    fn test_schedule_pi_over_8_z_rotation() {
        use BoardOccupancy::*;
        let width = 3_u32;
        let height = 4_u32;

        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 0, 2);
        let conf = Configuration {
            width,
            height,
            code_distance: 3,
            magic_state_distillation_cost: 5,
            num_distillations_for_pi_over_8_rotation: 1,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);
        assert!(!board.schedule_single_qubit_pi_over_8_rotation(Qubit::new(0), Pauli::Z, 5, 2));

        board.cycle = 5;
        assert!(!board.schedule_single_qubit_pi_over_8_rotation(Qubit::new(0), Pauli::Z, 5, 2));

        board.cycle = 10;
        assert!(!board.schedule_single_qubit_pi_over_8_rotation(Qubit::new(0), Pauli::Z, 5, 2));

        assert!(board.is_occupancy(0, 0, 0..16, Vacant));
        assert!(board.is_occupancy(0, 1, 0..16, IdleDataQubit));
        assert!(board.is_occupancy(0, 2, 0..16, IdleDataQubit));
        assert!(board.is_occupancy(0, 3, 0..16, Vacant));
        assert!(board.is_occupancy(1, 0, 0..16, Vacant));
        assert!(board.is_occupancy(1, 1, 0..16, Vacant));
        assert!(board.is_occupancy(1, 2, 0..16, Vacant));
        assert!(board.is_occupancy(1, 3, 0..16, Vacant));
        assert!(board.is_occupancy(2, 0, 0..16, Vacant));
        assert!(board.is_occupancy(2, 1, 0..16, Vacant));
        assert!(board.is_occupancy(2, 2, 0..16, Vacant));
        assert!(board.is_occupancy(2, 3, 0..16, Vacant));

        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 0));
        assert!(!board.has_schedule_at_or_after(Qubit::new(1), 0));

        board.cycle = 15;
        assert!(board.schedule_single_qubit_pi_over_8_rotation(Qubit::new(0), Pauli::Z, 5, 2));
        let id = OperationId { id: 0 };

        assert!(board.is_occupancy(0, 0, 0..5, Vacant));
        assert!(board.is_occupancy(0, 0, 5..15, MagicStateDistillation(id)));
        assert!(board.is_occupancy(0, 0, 15..18, LatticeSurgery(id)));
        assert!(board.is_occupancy(0, 0, 18..21, YMeasurement(id)));
        assert!(board.is_occupancy(0, 1, 0..15, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 15..18, DataQubitInOperation(id)));
        assert!(board.is_occupancy(0, 1, 18..21, IdleDataQubit));
        assert!(board.is_occupancy(0, 2, 0..21, IdleDataQubit));
        assert!(board.is_occupancy(0, 3, 0..21, Vacant));
        assert!(board.is_occupancy(1, 0, 0..5, Vacant));
        assert!(board.is_occupancy(1, 0, 5..15, MagicStateDistillation(id)));
        assert!(board.is_occupancy(1, 0, 15..18, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 0, 18..21, YMeasurement(id)));
        assert!(board.is_occupancy(1, 1, 0..21, Vacant));
        assert!(board.is_occupancy(1, 2, 0..21, Vacant));
        assert!(board.is_occupancy(1, 3, 0..21, Vacant));
        assert!(board.is_occupancy(2, 0, 0..21, Vacant));
        assert!(board.is_occupancy(2, 1, 0..21, Vacant));
        assert!(board.is_occupancy(2, 2, 0..21, Vacant));
        assert!(board.is_occupancy(2, 3, 0..21, Vacant));

        assert!(board.has_schedule_at_or_after(Qubit::new(0), 17));
        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 18));
        assert!(!board.has_schedule_at_or_after(Qubit::new(1), 0));

        let operations = vec![OperationWithAdditionalData::PiOver8Rotation {
            id,
            targets: vec![(Position { x: 0, y: 1 }, Pauli::Z)],
            routing_qubits: vec![],
            distillation_qubits: vec![Position { x: 0, y: 0 }, Position { x: 1, y: 0 }],
            num_distillations: 6,
            num_distillations_on_retry: 2,
        }];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_pi_over_8_x_rotation() {
        use BoardOccupancy::*;
        let width = 3_u32;
        let height = 4_u32;

        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 0, 2);
        let conf = Configuration {
            width,
            height,
            code_distance: 3,
            magic_state_distillation_cost: 5,
            num_distillations_for_pi_over_8_rotation: 1,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);
        assert!(!board.schedule_single_qubit_pi_over_8_rotation(Qubit::new(0), Pauli::X, 5, 2));

        board.cycle = 5;
        assert!(!board.schedule_single_qubit_pi_over_8_rotation(Qubit::new(0), Pauli::X, 5, 2));

        board.cycle = 10;
        assert!(!board.schedule_single_qubit_pi_over_8_rotation(Qubit::new(0), Pauli::X, 5, 2));

        assert!(board.is_occupancy(0, 0, 0..16, Vacant));
        assert!(board.is_occupancy(0, 1, 0..16, IdleDataQubit));
        assert!(board.is_occupancy(0, 2, 0..16, IdleDataQubit));
        assert!(board.is_occupancy(0, 3, 0..16, Vacant));
        assert!(board.is_occupancy(1, 0, 0..16, Vacant));
        assert!(board.is_occupancy(1, 1, 0..16, Vacant));
        assert!(board.is_occupancy(1, 2, 0..16, Vacant));
        assert!(board.is_occupancy(1, 3, 0..16, Vacant));
        assert!(board.is_occupancy(2, 0, 0..16, Vacant));
        assert!(board.is_occupancy(2, 1, 0..16, Vacant));
        assert!(board.is_occupancy(2, 2, 0..16, Vacant));
        assert!(board.is_occupancy(2, 3, 0..16, Vacant));

        board.cycle = 15;
        assert!(board.schedule_single_qubit_pi_over_8_rotation(Qubit::new(0), Pauli::X, 5, 2));

        let id = OperationId { id: 0 };

        assert!(board.is_occupancy(0, 0, 0..21, Vacant));
        assert!(board.is_occupancy(0, 1, 0..15, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 15..18, DataQubitInOperation(id)));
        assert!(board.is_occupancy(0, 1, 18..21, IdleDataQubit));
        assert!(board.is_occupancy(0, 2, 0..21, IdleDataQubit));
        assert!(board.is_occupancy(0, 3, 0..21, Vacant));
        assert!(board.is_occupancy(1, 0, 0..21, Vacant));
        assert!(board.is_occupancy(1, 1, 0..5, Vacant));
        assert!(board.is_occupancy(1, 1, 5..15, MagicStateDistillation(id)));
        assert!(board.is_occupancy(1, 1, 15..18, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 1, 18..21, YMeasurement(id)));
        assert!(board.is_occupancy(1, 2, 0..21, Vacant));
        assert!(board.is_occupancy(1, 3, 0..21, Vacant));
        assert!(board.is_occupancy(2, 0, 0..21, Vacant));
        assert!(board.is_occupancy(2, 1, 0..5, Vacant));
        assert!(board.is_occupancy(2, 1, 5..15, MagicStateDistillation(id)));
        assert!(board.is_occupancy(2, 1, 15..18, LatticeSurgery(id)));
        assert!(board.is_occupancy(2, 1, 18..21, YMeasurement(id)));
        assert!(board.is_occupancy(2, 2, 0..21, Vacant));
        assert!(board.is_occupancy(2, 3, 0..21, Vacant));
        assert!(board.has_schedule_at_or_after(Qubit::new(0), 17));
        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 18));
        assert!(!board.has_schedule_at_or_after(Qubit::new(1), 0));

        let operations = vec![OperationWithAdditionalData::PiOver8Rotation {
            id,
            targets: vec![(Position { x: 0, y: 1 }, Pauli::X)],
            routing_qubits: vec![],
            distillation_qubits: vec![Position { x: 1, y: 1 }, Position { x: 2, y: 1 }],
            num_distillations: 6,
            num_distillations_on_retry: 2,
        }];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_pi_over_8_y_rotation() {
        use BoardOccupancy::*;
        let width = 3_u32;
        let height = 4_u32;

        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 1);
        mapping.map(Qubit::new(1), 0, 2);
        let conf = Configuration {
            width,
            height,
            code_distance: 3,
            magic_state_distillation_cost: 5,
            num_distillations_for_pi_over_8_rotation: 5,
            preferable_distillation_area_size: 3,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);
        assert!(!board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver8,
            axis: new_axis("YI")
        }));

        board.cycle = 5;
        assert!(!board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver8,
            axis: new_axis("YI")
        }));

        assert!(board.is_occupancy(0, 0, 0..8, Vacant));
        assert!(board.is_occupancy(0, 1, 0..8, IdleDataQubit));
        assert!(board.is_occupancy(0, 2, 0..8, IdleDataQubit));
        assert!(board.is_occupancy(0, 3, 0..8, Vacant));
        assert!(board.is_occupancy(1, 0, 0..8, Vacant));
        assert!(board.is_occupancy(1, 1, 0..8, Vacant));
        assert!(board.is_occupancy(1, 2, 0..8, Vacant));
        assert!(board.is_occupancy(1, 3, 0..8, Vacant));
        assert!(board.is_occupancy(2, 0, 0..8, Vacant));
        assert!(board.is_occupancy(2, 1, 0..8, Vacant));
        assert!(board.is_occupancy(2, 2, 0..8, Vacant));
        assert!(board.is_occupancy(2, 3, 0..8, Vacant));

        board.cycle = 10;
        assert!(board.schedule_rotation(&PauliRotation {
            angle: Angle::PiOver8,
            axis: new_axis("YI")
        }));

        let id = OperationId { id: 0 };

        assert!(board.is_occupancy(0, 0, 0..10, Vacant));
        assert!(board.is_occupancy(0, 0, 10..13, LatticeSurgery(id)));
        assert!(board.is_occupancy(0, 0, 13..16, Vacant));
        assert!(board.is_occupancy(0, 1, 0..10, IdleDataQubit));
        assert!(board.is_occupancy(0, 1, 10..13, DataQubitInOperation(id)));
        assert!(board.is_occupancy(0, 1, 13..16, IdleDataQubit));
        assert!(board.is_occupancy(0, 2, 0..16, IdleDataQubit));
        assert!(board.is_occupancy(0, 3, 0..16, Vacant));
        assert!(board.is_occupancy(1, 0, 0..10, Vacant));
        assert!(board.is_occupancy(1, 0, 10..13, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 0, 13..16, Vacant));
        assert!(board.is_occupancy(1, 1, 0..10, Vacant));
        assert!(board.is_occupancy(1, 1, 10..13, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 1, 13..16, Vacant));
        assert!(board.is_occupancy(1, 2, 0..10, MagicStateDistillation(id)));
        assert!(board.is_occupancy(1, 2, 10..13, LatticeSurgery(id)));
        assert!(board.is_occupancy(1, 2, 13..16, YMeasurement(id)));
        assert!(board.is_occupancy(1, 3, 0..16, Vacant));
        assert!(board.is_occupancy(2, 0, 0..16, Vacant));
        assert!(board.is_occupancy(2, 1, 0..10, MagicStateDistillation(id)));
        assert!(board.is_occupancy(2, 1, 10..13, LatticeSurgery(id)));
        assert!(board.is_occupancy(2, 1, 13..16, YMeasurement(id)));
        assert!(board.is_occupancy(2, 2, 0..10, MagicStateDistillation(id)));
        assert!(board.is_occupancy(2, 2, 10..13, LatticeSurgery(id)));
        assert!(board.is_occupancy(2, 2, 13..16, YMeasurement(id)));
        assert!(board.is_occupancy(2, 3, 0..16, Vacant));
        assert!(board.has_schedule_at_or_after(Qubit::new(0), 12));
        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 13));
        assert!(!board.has_schedule_at_or_after(Qubit::new(1), 0));

        let operations = vec![OperationWithAdditionalData::PiOver8Rotation {
            id,
            targets: vec![(Position { x: 0, y: 1 }, Pauli::Y)],
            routing_qubits: vec![
                Position { x: 1, y: 1 },
                Position { x: 1, y: 0 },
                Position { x: 0, y: 0 },
            ],
            distillation_qubits: vec![
                Position { x: 2, y: 1 },
                Position { x: 2, y: 2 },
                Position { x: 1, y: 2 },
            ],
            num_distillations: 6,
            num_distillations_on_retry: 3,
        }];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_schedule_pi_over_8_rotation_block_target_will_be_used() {
        let width = 3_u32;
        let height = 4_u32;

        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 1);
        let conf = Configuration {
            width,
            height,
            code_distance: 3,
            magic_state_distillation_cost: 5,
            num_distillations_for_pi_over_8_rotation: 5,
            num_distillations_for_pi_over_8_rotation_block: 5,
            single_qubit_pi_over_8_rotation_block_depth_ratio: 1.5,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);

        board.set_cycle_after_last_operation_at(Qubit::new(0), 1000);
        assert!(!board.schedule_single_qubit_pi_over_8_rotation_block(
            Qubit::new(0),
            &[Pauli::X, Pauli::Y, Pauli::Z],
            &[Pauli::X, Pauli::Y],
        ));
        assert!(board.operations.is_empty());
    }

    #[test]
    fn test_schedule_pi_over_8_rotation_block_routing_qubits_are_unavailable() {
        let width = 3_u32;
        let height = 4_u32;

        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 1);
        let conf = Configuration {
            width,
            height,
            code_distance: 3,
            magic_state_distillation_cost: 5,
            num_distillations_for_pi_over_8_rotation_block: 5,
            single_qubit_pi_over_8_rotation_block_depth_ratio: 1.2,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);
        let id = board.issue_operation_id();
        board.ensure_board_occupancy(1);
        board.set_occupancy(0, 0, 0, BoardOccupancy::LatticeSurgery(id));
        board.set_occupancy(2, 0, 0, BoardOccupancy::LatticeSurgery(id));
        board.set_occupancy(0, 2, 0, BoardOccupancy::LatticeSurgery(id));
        board.set_occupancy(2, 2, 0, BoardOccupancy::LatticeSurgery(id));

        assert!(!board.schedule_single_qubit_pi_over_8_rotation_block(
            Qubit::new(0),
            &[Pauli::X, Pauli::Z, Pauli::X],
            &[Pauli::X, Pauli::Y],
        ));
        assert!(board.operations.is_empty());
    }

    #[test]
    fn test_schedule_pi_over_8_rotation_distillation_block_qubits_are_unavailable() {
        let width = 2_u32;
        let height = 4_u32;

        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 0);
        let conf = Configuration {
            width,
            height,
            code_distance: 3,
            magic_state_distillation_cost: 5,
            num_distillations_for_pi_over_8_rotation_block: 5,
            single_qubit_pi_over_8_rotation_block_depth_ratio: 1.2,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);

        assert!(!board.schedule_single_qubit_pi_over_8_rotation_block(
            Qubit::new(0),
            &[Pauli::X, Pauli::Z, Pauli::X],
            &[Pauli::X, Pauli::Y],
        ));
        assert!(board.operations.is_empty());
    }

    #[test]
    fn test_schedule_pi_over_8_rotation_distillation_block() {
        use BoardOccupancy::*;
        let width = 3_u32;
        let height = 4_u32;

        let mut mapping = DataQubitMapping::new(width, height);
        mapping.map(Qubit::new(0), 0, 0);
        let conf = Configuration {
            width,
            height,
            code_distance: 3,
            magic_state_distillation_cost: 5,
            magic_state_distillation_success_rate: 0.2,
            num_distillations_for_pi_over_8_rotation_block: 5,
            single_qubit_pi_over_8_rotation_block_depth_ratio: 1.5,
            ..default_conf()
        };
        let mut board = Board::new(mapping, &conf);
        let pi_over_8_axes = [Pauli::X, Pauli::Z, Pauli::X];
        let pi_over_4_axes = [Pauli::X, Pauli::Y];

        assert!(board.schedule_single_qubit_pi_over_8_rotation_block(
            Qubit::new(0),
            &pi_over_8_axes,
            &pi_over_4_axes,
        ));

        let id = OperationId { id: 0 };

        assert!(board.is_occupancy(0, 0, 0..33, DataQubitInOperation(id)));
        assert!(board.is_occupancy(0, 0, 33..40, IdleDataQubit));
        assert!(board.is_occupancy(1, 0, 0..36, PiOver8RotationBlock(id)));
        assert!(board.is_occupancy(1, 0, 36..40, Vacant));
        assert!(board.is_occupancy(2, 0, 0..30, PiOver8RotationBlock(id)));
        assert!(board.is_occupancy(2, 0, 30..40, Vacant));

        assert!(board.is_occupancy(0, 1, 0..36, PiOver8RotationBlock(id)));
        assert!(board.is_occupancy(0, 1, 36..40, Vacant));
        assert!(board.is_occupancy(1, 1, 0..33, PiOver8RotationBlock(id)));
        assert!(board.is_occupancy(1, 1, 33..40, Vacant));
        assert!(board.is_occupancy(2, 1, 0..30, PiOver8RotationBlock(id)));
        assert!(board.is_occupancy(2, 1, 30..40, Vacant));

        assert!(board.is_occupancy(0, 2, 0..33, PiOver8RotationBlock(id)));
        assert!(board.is_occupancy(0, 2, 33..40, Vacant));
        assert!(board.is_occupancy(1, 2, 0..30, PiOver8RotationBlock(id)));
        assert!(board.is_occupancy(1, 2, 30..40, Vacant));
        assert!(board.is_occupancy(2, 2, 0..40, Vacant));

        assert!(board.is_occupancy(0, 3, 0..30, PiOver8RotationBlock(id)));
        assert!(board.is_occupancy(0, 3, 30..40, Vacant));
        assert!(board.is_occupancy(1, 3, 0..40, Vacant));
        assert!(board.is_occupancy(2, 3, 0..40, Vacant));

        assert!(board.has_schedule_at_or_after(Qubit::new(0), 32));
        assert!(!board.has_schedule_at_or_after(Qubit::new(0), 33));

        let operations = vec![
            OperationWithAdditionalData::SingleQubitPiOver8RotationBlock {
                id,
                target: Position { x: 0, y: 0 },
                routing_qubits: vec![p(1, 0), p(0, 1), p(1, 1)],
                distillation_qubits: vec![p(0, 2), p(0, 3), p(1, 2), p(2, 0), p(2, 1)],
                correction_qubits: vec![p(0, 2), p(1, 0), p(0, 1)],
                pi_over_8_axes: pi_over_8_axes.to_vec(),
                pi_over_4_axes: pi_over_4_axes.to_vec(),
            },
        ];
        assert_eq!(board.operations, operations);
    }

    #[test]
    fn test_pi_over_4_rotation_serialization() {
        let id = OperationId { id: 99 };
        let pos1 = Position { x: 0, y: 1 };
        let pos2 = Position { x: 1, y: 3 };
        let targets = vec![(pos1, Pauli::Z), (pos2, Pauli::X)];
        let ancilla_qubits = vec![
            Position { x: 0, y: 2 },
            Position { x: 0, y: 3 },
            Position { x: 0, y: 4 },
        ];
        let op = OperationWithAdditionalData::PiOver4Rotation {
            id,
            targets,
            ancilla_qubits,
        };

        let serialized = serde_json::to_string(&op).unwrap();
        let expectation = r#"{"type":"PI_OVER_4_ROTATION","id":99,"targets":[{"x":0,"y":1,"axis":"Z"},{"x":1,"y":3,"axis":"X"}],"ancilla_qubits":[{"x":0,"y":2},{"x":0,"y":3},{"x":0,"y":4}]}"#;

        assert_eq!(serialized, expectation);
    }

    #[test]
    fn test_pi_over_8_rotation_serialization() {
        let id = OperationId { id: 91 };
        let pos1 = Position { x: 0, y: 1 };
        let pos2 = Position { x: 0, y: 3 };
        let targets = vec![(pos1, Pauli::Y), (pos2, Pauli::X)];
        let routing_qubits = vec![Position { x: 1, y: 1 }];
        let distillation_qubits = vec![Position { x: 1, y: 2 }];
        let op = OperationWithAdditionalData::PiOver8Rotation {
            id,
            targets,
            routing_qubits,
            distillation_qubits,
            num_distillations: 4,
            num_distillations_on_retry: 2,
        };

        let serialized = serde_json::to_string(&op).unwrap();
        let expectation = r#"
        {
            "type":"PI_OVER_8_ROTATION",
            "id":91,
            "targets":[{"x":0,"y":1,"axis":"Y"},{"x":0,"y":3,"axis":"X"}],
            "routing_qubits":[{"x":1,"y":1}],
            "distillation_qubits":[{"x":1,"y":2}],
            "num_distillations":4,
            "num_distillations_on_retry":2
        }"#
        .replace("\n", "")
        .replace(" ", "");

        assert_eq!(serialized, expectation);
    }

    #[test]
    fn test_single_qubit_arbitrary_angle_rotation_serialization() {
        let id = OperationId { id: 91 };
        let pos = Position { x: 0, y: 2 };
        let pi_over_8_axes = [Pauli::Y, Pauli::Z, Pauli::X, Pauli::Z];
        let pi_over_4_axes = [Pauli::X, Pauli::Y];
        let routing_qubits = [p(1, 2), p(0, 1), p(1, 1)];
        let distributions_qubits = [p(0, 0), p(1, 0)];
        let correction_qubits = [p(0, 0), p(1, 2), p(0, 1)];
        let op = OperationWithAdditionalData::SingleQubitPiOver8RotationBlock {
            id,
            target: pos,
            routing_qubits: routing_qubits.to_vec(),
            distillation_qubits: distributions_qubits.to_vec(),
            correction_qubits: correction_qubits.to_vec(),
            pi_over_8_axes: pi_over_8_axes.to_vec(),
            pi_over_4_axes: pi_over_4_axes.to_vec(),
        };

        let serialized = serde_json::to_string(&op).unwrap();
        let expectation = r#"
        {
            "type":"SINGLE_QUBIT_PI_OVER_8_ROTATION_BLOCK",
            "id":91,
            "target":{"x":0,"y":2},
            "routing_qubits": [{"x":1,"y":2},{"x":0,"y":1},{"x":1,"y":1}],
            "distillation_qubits": [{"x":0,"y":0},{"x":1,"y":0}],
            "correction_qubits": [{"x":0,"y":0},{"x":1,"y":2},{"x":0,"y":1}],
            "pi_over_8_axes":["Y","Z","X","Z"],
            "pi_over_4_axes":["X","Y"]
        }"#
        .replace("\n", "")
        .replace(" ", "");

        assert_eq!(serialized, expectation);
    }
}
