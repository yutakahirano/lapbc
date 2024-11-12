import mapping_generator
import unittest


class MappingTest(unittest.TestCase):
    def test_dense(self):
        mapping = mapping_generator.generate_dense(6)

        self.assertEqual(mapping.width, 10)
        self.assertEqual(mapping.height, 10)
        self.assertEqual(mapping.mapping, [
            {"x": 1, "y": 1, "qubit": 0},
            {"x": 2, "y": 1, "qubit": 1},
            {"x": 4, "y": 1, "qubit": 2},
            {"x": 5, "y": 1, "qubit": 3},
            {"x": 7, "y": 1, "qubit": 4},
            {"x": 8, "y": 1, "qubit": 5},
            {"x": 1, "y": 2, "qubit": 6},
            {"x": 2, "y": 2, "qubit": 7},
            {"x": 4, "y": 2, "qubit": 8},
            {"x": 5, "y": 2, "qubit": 9},
            {"x": 7, "y": 2, "qubit": 10},
            {"x": 8, "y": 2, "qubit": 11},

            {"x": 1, "y": 4, "qubit": 12},
            {"x": 2, "y": 4, "qubit": 13},
            {"x": 4, "y": 4, "qubit": 14},
            {"x": 5, "y": 4, "qubit": 15},
            {"x": 7, "y": 4, "qubit": 16},
            {"x": 8, "y": 4, "qubit": 17},
            {"x": 1, "y": 5, "qubit": 18},
            {"x": 2, "y": 5, "qubit": 19},
            {"x": 4, "y": 5, "qubit": 20},
            {"x": 5, "y": 5, "qubit": 21},
            {"x": 7, "y": 5, "qubit": 22},
            {"x": 8, "y": 5, "qubit": 23},

            {"x": 1, "y": 7, "qubit": 24},
            {"x": 2, "y": 7, "qubit": 25},
            {"x": 4, "y": 7, "qubit": 26},
            {"x": 5, "y": 7, "qubit": 27},
            {"x": 7, "y": 7, "qubit": 28},
            {"x": 8, "y": 7, "qubit": 29},
            {"x": 1, "y": 8, "qubit": 30},
            {"x": 2, "y": 8, "qubit": 31},
            {"x": 4, "y": 8, "qubit": 32},
            {"x": 5, "y": 8, "qubit": 33},
            {"x": 7, "y": 8, "qubit": 34},
            {"x": 8, "y": 8, "qubit": 35}
        ])

    def test_sparse(self):
        mapping = mapping_generator.generate_sparse(6)

        self.assertEqual(mapping.width, 13)
        self.assertEqual(mapping.height, 13)
        self.assertEqual(mapping.mapping, [
            {"x": 1, "y": 1, "qubit": 0},
            {"x": 3, "y": 1, "qubit": 1},
            {"x": 5, "y": 1, "qubit": 2},
            {"x": 7, "y": 1, "qubit": 3},
            {"x": 9, "y": 1, "qubit": 4},
            {"x": 11, "y": 1, "qubit": 5},

            {"x": 1, "y": 3, "qubit": 6},
            {"x": 3, "y": 3, "qubit": 7},
            {"x": 5, "y": 3, "qubit": 8},
            {"x": 7, "y": 3, "qubit": 9},
            {"x": 9, "y": 3, "qubit": 10},
            {"x": 11, "y": 3, "qubit": 11},

            {"x": 1, "y": 5, "qubit": 12},
            {"x": 3, "y": 5, "qubit": 13},
            {"x": 5, "y": 5, "qubit": 14},
            {"x": 7, "y": 5, "qubit": 15},
            {"x": 9, "y": 5, "qubit": 16},
            {"x": 11, "y": 5, "qubit": 17},

            {"x": 1, "y": 7, "qubit": 18},
            {"x": 3, "y": 7, "qubit": 19},
            {"x": 5, "y": 7, "qubit": 20},
            {"x": 7, "y": 7, "qubit": 21},
            {"x": 9, "y": 7, "qubit": 22},
            {"x": 11, "y": 7, "qubit": 23},

            {"x": 1, "y": 9, "qubit": 24},
            {"x": 3, "y": 9, "qubit": 25},
            {"x": 5, "y": 9, "qubit": 26},
            {"x": 7, "y": 9, "qubit": 27},
            {"x": 9, "y": 9, "qubit": 28},
            {"x": 11, "y": 9, "qubit": 29},

            {"x": 1, "y": 11, "qubit": 30},
            {"x": 3, "y": 11, "qubit": 31},
            {"x": 5, "y": 11, "qubit": 32},
            {"x": 7, "y": 11, "qubit": 33},
            {"x": 9, "y": 11, "qubit": 34},
            {"x": 11, "y": 11, "qubit": 35}
        ])

if __name__ == '__main__':
    unittest.main()