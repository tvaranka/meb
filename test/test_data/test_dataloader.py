import unittest
from meb.datasets import only_digit


class TestDatasetUtils(unittest.TestCase):
    def test_only_digit(self):
        test_strings = ["AU12", "AU5R", "1L", "4 B", "6(k)"]
        test_results = ["12", "5", "1", "4", "6"]
        for string, result in zip(test_strings, test_results):
            digit = only_digit(string)
            self.assertEqual(digit, result)


if __name__ == '__main__':
    unittest.main()

