import enum
import json
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np

from enum import Enum, auto
from matplotlib.animation import FuncAnimation
from typing import Iterable


class Pauli(Enum):
    IDENTITY = auto(),  # pycodestyle complains about 'I'.
    X = auto(),
    Y = auto(),
    Z = auto()

    @staticmethod
    def from_string(name: str):
        match name:
            case 'I':
                return Pauli.IDENTITY
            case 'X':
                return Pauli.X
            case 'Y':
                return Pauli.Y
            case 'Z':
                return Pauli.Z
            case _:
                raise ValueError(f'Unknown Pauli: {name}')


class Vacant:
    def __str__(self) -> str:
        return 'Vacant'

    @staticmethod
    def class_id() -> int:
        return 1

    def type_id(self) -> int:
        return Vacant.class_id()


class LatticeSurgery:
    def __init__(self, id: int):
        self.id = id

    def __str__(self) -> str:
        return f'LatticeSurgery({self.id})'

    @staticmethod
    def class_id() -> int:
        return 2

    def type_id(self) -> int:
        return LatticeSurgery.class_id()


class IdleDataQubit:
    def __str__(self) -> str:
        return 'IdleDataQubit'

    @staticmethod
    def class_id() -> int:
        return 3

    def type_id(self) -> int:
        return IdleDataQubit.class_id()


class DataQubitInOperation:
    def __init__(self, id: int):
        self.id = id

    def __str__(self) -> str:
        return f'DataQubitInOperation({self.id})'

    @staticmethod
    def class_id() -> int:
        return 4

    def type_id(self) -> int:
        return DataQubitInOperation.class_id()


class YInitialization:
    def __init__(self, id: int):
        self.id = id

    def __str__(self) -> str:
        return 'YInitialization'

    @staticmethod
    def class_id() -> int:
        return 5

    def type_id(self) -> int:
        return YInitialization.class_id()


class YMeasurement:
    def __init__(self, id: int):
        self.id = id

    def __str__(self) -> str:
        return 'YMeasurement'

    @staticmethod
    def class_id() -> int:
        return 6

    def type_id(self) -> int:
        return YMeasurement.class_id()


class MagicStateDistillation:
    def __init__(self, id: int):
        self.id = id

    def __str__(self) -> str:
        format = 'MagicStateDistillation(id={})'
        return format.format(self.id)

    @staticmethod
    def class_id() -> int:
        return 7

    def type_id(self) -> int:
        return MagicStateDistillation.class_id()


class PiOver8RotationBlock:
    def __init__(self, id: int):
        self.id = id

    def __str__(self) -> str:
        format = 'PiOver8RotationBlock(id={})'
        return format.format(self.id)

    @staticmethod
    def class_id() -> int:
        return 8

    def type_id(self) -> int:
        return PiOver8RotationBlock.class_id()


Occupancy = Vacant | LatticeSurgery | IdleDataQubit | DataQubitInOperation | \
            YInitialization | YMeasurement | MagicStateDistillation | \
            PiOver8RotationBlock


class PiOver4Rotation:
    def __init__(self, id: int, targets: list[tuple[int, int, Pauli]]):
        self.id = id
        self.targets = targets


class PiOver8Rotation:
    def __init__(self, id: int, targets: list[tuple[int, int, Pauli]],
                 num_distillations: int, num_distillations_on_retry: int):
        self.id = id
        self.targets = targets
        self.num_distillations = num_distillations
        self.num_distillations_on_retry = num_distillations_on_retry


class SingleQubitPiOver8RotationBlock:
    def __init__(self, id: int, target: tuple[int, int],
                 pi_over_8_rotation_axes: list[Pauli], pi_over_4_rotation_axes: list[Pauli]):
        self.id = id
        self.target = target
        self.pi_over_8_rotation_axes = pi_over_8_rotation_axes
        self.pi_over_4_rotation_axes = pi_over_4_rotation_axes


Operation = PiOver4Rotation | PiOver8Rotation | SingleQubitPiOver8RotationBlock


class ScheduleEntry:
    def __init__(self, x: int, y: int, occupancy: Occupancy):
        self.x = x
        self.y = y
        self.occupancy = occupancy

    def __str__(self) -> str:
        return f'ScheduledEntry(x={self.x}, y={self.y}, occupancy={self.occupancy})'


class Schedule:
    def __init__(self, width: int, height: int, operations: dict[int, Operation], schedule: list[list[ScheduleEntry]]):
        self.width = width
        self.height = height
        self.operations = operations
        self.schedule = schedule


def load_schedule(path: str) -> Schedule:
    with open(path, 'r') as f:
        data = json.loads(f.read())

    assert 'width' in data
    width = int(data['width'])
    assert 'height' in data
    height = int(data['height'])
    assert 'operations' in data
    json_operations = data['operations']
    assert isinstance(json_operations, list)
    assert 'schedule' in data
    json_schedule = data['schedule']
    assert isinstance(json_schedule, list)
    cycle = 0
    operations: dict[int, Operation] = {}
    for json_operation in json_operations:
        assert isinstance(json_operation, dict)
        assert 'id' in json_operation
        operation_id = json_operation['id']
        assert 'type' in json_operation
        typename = json_operation['type']
        assert 'id' in json_operation
        operation_id = int(json_operation['id'])

        match typename:
            case 'PI_OVER_4_ROTATION':
                assert 'targets' in json_operation
                json_targets = json_operation['targets']
                targets: list[tuple[int, int, Pauli]] = []
                assert isinstance(json_targets, list)
                for json_target in json_targets:
                    assert isinstance(json_target, dict)
                    assert 'x' in json_target
                    x = int(json_target['x'])
                    assert 'y' in json_target
                    y = int(json_target['y'])
                    assert 'axis' in json_target
                    axis = Pauli.from_string(json_target['axis'])
                    targets.append((x, y, axis))
                operations[operation_id] = PiOver4Rotation(operation_id, targets)
            case 'PI_OVER_8_ROTATION':
                assert 'targets' in json_operation
                json_targets = json_operation['targets']
                targets = []
                assert isinstance(json_targets, list)
                for json_target in json_targets:
                    assert isinstance(json_target, dict)
                    assert 'x' in json_target
                    x = int(json_target['x'])
                    assert 'y' in json_target
                    y = int(json_target['y'])
                    assert 'axis' in json_target
                    axis = Pauli.from_string(json_target['axis'])
                    targets.append((x, y, axis))
                assert 'num_distillations' in json_operation
                num_distillations = int(json_operation['num_distillations'])
                assert 'num_distillations_on_retry' in json_operation
                num_distillations_on_retry = int(json_operation['num_distillations_on_retry'])
                operations[operation_id] = PiOver8Rotation(
                    operation_id, targets, num_distillations, num_distillations_on_retry)
            case 'SINGLE_QUBIT_PI_OVER_8_ROTATION_BLOCK':
                assert 'target' in json_operation
                json_target = json_operation['target']
                assert isinstance(json_target, dict)
                assert 'x' in json_target
                x = int(json_target['x'])
                assert 'y' in json_target
                y = int(json_target['y'])
                assert 'pi_over_8_axes' in json_operation
                json_pi_over_8_rotation_axes = json_operation['pi_over_8_axes']
                pi_over_8_rotation_axes: list[Pauli] = []
                assert isinstance(json_pi_over_8_rotation_axes, list)
                for json_axis in json_pi_over_8_rotation_axes:
                    pi_over_8_rotation_axes.append(Pauli.from_string(json_axis))
                assert 'pi_over_4_axes' in json_operation
                json_pi_over_4_rotation_axes = json_operation['pi_over_4_axes']
                pi_over_4_rotation_axes: list[Pauli] = []
                assert isinstance(json_pi_over_4_rotation_axes, list)
                for json_axis in json_pi_over_4_rotation_axes:
                    pi_over_4_rotation_axes.append(Pauli.from_string(json_axis))
                operations[operation_id] = SingleQubitPiOver8RotationBlock(
                    operation_id, (x, y), pi_over_8_rotation_axes, pi_over_4_rotation_axes)
            case _:
                raise ValueError(f'Unknown operation type: {typename}')

    entries: list[list[ScheduleEntry]] = []
    for json_schedule_at_cycle in json_schedule:
        entries_at_cycle: list[ScheduleEntry] = []
        for json_entry in json_schedule_at_cycle:
            assert isinstance(json_entry, dict)
            assert 'x' in json_entry
            x = int(json_entry['x'])
            assert 'y' in json_entry
            y = int(json_entry['y'])
            assert 'occupancy' in json_entry
            occupancy: Occupancy
            match json_entry['occupancy']:
                case {'type': 'VACANT'}:
                    occupancy = Vacant()
                case {'type': 'LATTICE_SURGERY', 'operation_id': operation_id}:
                    occupancy = LatticeSurgery(int(operation_id))
                case {'type': 'IDLE_DATA_QUBIT'}:
                    occupancy = IdleDataQubit()
                case {'type': 'DATA_QUBIT_IN_OPERATION', 'operation_id': operation_id}:
                    occupancy = DataQubitInOperation(int(operation_id))
                case {'type': 'Y_INITIALIZATION', 'operation_id': operation_id}:
                    occupancy = YInitialization(int(operation_id))
                case {'type': 'Y_MEASUREMENT', 'operation_id': operation_id}:
                    occupancy = YMeasurement(int(operation_id))
                case {'type': 'MAGIC_STATE_DISTILLATION', 'operation_id': operation_id}:
                    occupancy = MagicStateDistillation(int(operation_id))
                case {'type': 'PI_OVER_8_ROTATION_BLOCK', 'operation_id': operation_id}:
                    occupancy = PiOver8RotationBlock(int(operation_id))
            entry = ScheduleEntry(x, y, occupancy)
            entries_at_cycle.append(entry)

        entries.append(entries_at_cycle)
        cycle += 1
    return Schedule(width, height, operations, entries)


def create_normalizer_and_color_map() -> tuple[matplotlib.colors.Normalize, matplotlib.colors.LinearSegmentedColormap]:
    type_ids = [
        Vacant.class_id(),
        IdleDataQubit.class_id(),
        DataQubitInOperation.class_id(),
        LatticeSurgery.class_id(),
        YInitialization.class_id(),
        YMeasurement.class_id(),
        MagicStateDistillation.class_id(),
        PiOver8RotationBlock.class_id(),
    ]
    assert 0 not in type_ids
    vmin = min(type_ids)
    vmax = max(type_ids)
    assert vmin < vmax

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    norm_dict = {id: (id - vmin) / (vmax - vmin) for id in type_ids}

    clist: list[tuple[float, tuple[float, float, float]]] = [
        (norm_dict[Vacant.class_id()], (0xee, 0xee, 0xee)),
        (norm_dict[IdleDataQubit.class_id()], (0xcd, 0xda, 0xf6)),
        (norm_dict[DataQubitInOperation.class_id()], (0xb1, 0xa8, 0xd3)),
        (norm_dict[LatticeSurgery.class_id()], (0x7c, 0xa5, 0x58)),
        (norm_dict[YInitialization.class_id()], (0xcc, 0x6f, 0x68)),
        (norm_dict[YMeasurement.class_id()], (0xcc, 0x6f, 0x68)),
        (norm_dict[MagicStateDistillation.class_id()], (0x2f, 0x57, 0xc3)),
        (norm_dict[PiOver8RotationBlock.class_id()], (0xff, 0x99, 0x00)),
    ]
    clist.sort()
    for i in range(len(clist)):
        color = clist[i][1]
        clist[i] = (clist[i][0], (color[0] / 255, color[1] / 255, color[2] / 255))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', clist)
    return (norm, cmap)


def visualize(schedule: Schedule, path_pattern: str, cycle_range: Iterable[int] | None = None):
    if cycle_range is None:
        cycle_range = range(len(schedule.schedule))
    ext = 'png'

    norm, cmap = create_normalizer_and_color_map()
    previous_data: list[list[int]] | None = None

    for cycle in cycle_range:
        n = round(math.log10(len(schedule.schedule))) + 1
        path = '{}-{}.{}'.format(path_pattern, str(cycle).zfill(n), ext)

        entries: list[ScheduleEntry] = schedule.schedule[cycle]
        data: list[list[int]] = [[0 for x in range(schedule.width)] for y in range(schedule.height)]

        for entry in entries:
            occupancy = entry.occupancy
            x = entry.x
            y = entry.y
            data[x][y] = occupancy.type_id()

        if previous_data == data:
            continue
        previous_data = data

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        for entry in entries:
            occupancy = entry.occupancy
            x = entry.x
            y = entry.y
            if isinstance(occupancy, LatticeSurgery) or isinstance(occupancy, DataQubitInOperation) or \
               isinstance(occupancy, MagicStateDistillation) or isinstance(occupancy, YInitialization) or \
               isinstance(occupancy, YMeasurement) or isinstance(occupancy, PiOver8RotationBlock):
                operation_id = occupancy.id
                text_operation_id = '{:02x}'.format(operation_id & 0xff)
                ax.text(y, x, text_operation_id, ha='center', va='center')

        ax.imshow(data, norm=norm, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(path, transparent=True)
        plt.close()


def generate_animation(schedule: Schedule, path: str, cycle_range: Iterable[int] | None = None):
    if cycle_range is None:
        cycle_range = range(len(schedule.schedule))

    norm, cmap = create_normalizer_and_color_map()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    def animate(cycle: int):
        ax.clear()
        entries: list[ScheduleEntry] = schedule.schedule[cycle]
        data: list[list[int]] = [[0 for x in range(schedule.width)] for y in range(schedule.height)]
        for entry in entries:
            occupancy = entry.occupancy
            x = entry.x
            y = entry.y
            data[x][y] = occupancy.type_id()
        for entry in entries:
            occupancy = entry.occupancy
            x = entry.x
            y = entry.y
            if isinstance(occupancy, LatticeSurgery) or isinstance(occupancy, DataQubitInOperation) or \
               isinstance(occupancy, MagicStateDistillation) or isinstance(occupancy, YInitialization) or \
               isinstance(occupancy, YMeasurement) or isinstance(occupancy, PiOver8RotationBlock):
                operation_id = occupancy.id
                text_operation_id = '{:02x}'.format(operation_id & 0xff)
                ax.text(y, x, text_operation_id, ha='center', va='center')

        ax.set_title('Cycle {}'.format(cycle))
        ax.imshow(data, norm=norm, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])

    animation = FuncAnimation(fig, animate, frames=cycle_range, repeat=False, interval=200)
    animation.save(path, writer='pillow', fps=5)
