import argparse
import json


class Mapping:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.mapping: list[dict[str, int]] = []
    
    def add(self, x: int, y: int, qubit: int) -> None:
        self.mapping.append({'x': x, 'y': y, 'qubit': qubit})

    def to_json(self) -> str:
        return json.dumps({
            'width': self.width,
            'height': self.height,
            'mapping': self.mapping
        }, sort_keys=True, indent=2)


def generate_dense(width: int) -> Mapping:
    assert width % 2 == 0
    mapping_width = width + (width // 2) + 1
    mapping = Mapping(mapping_width, mapping_width)

    q = 0
    for i in range(width // 2):
        y = i * 2 + i + 1
        for j in range(width // 2):
            x = j * 2 + j + 1
            mapping.add(x, y, q)
            mapping.add(x + 1, y, q + 1)
            q += 2
        for j in range(width // 2):
            x = j * 2 + j + 1
            mapping.add(x, y + 1, q)
            mapping.add(x + 1, y + 1, q + 1)
            q += 2

    return mapping


def generate_sparse(width: int) -> Mapping:
    assert width % 2 == 0
    mapping_width = width * 2 + 1
    mapping = Mapping(mapping_width, mapping_width)

    q = 0
    for i in range(width):
        for j in range(width):
            y = i * 2 + 1
            x = j * 2 + 1
            mapping.add(x, y, q)
            q += 1

    return mapping

def main() -> None:
    parser = argparse.ArgumentParser(
        description='description')
    parser.add_argument('--width', type=int, default=6)
    parser.add_argument('--sparse', action='store_true')
    args = parser.parse_args()

    if args.width % 2 != 0:
        raise ValueError('width must be even')
    
    if args.sparse:
        mapping: Mapping = generate_sparse(args.width)
    else:
        mapping = generate_dense(args.width)
    
    print(mapping.to_json())


if __name__ == '__main__':
    main()

