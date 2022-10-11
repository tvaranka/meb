import unittest
import pandas as pd
import numpy as np
from meb.datasets.dataset_utils import extract_action_units


class TestDatasetUtils(unittest.TestCase):
    def test_extract_action_units(self):
        test_df = pd.DataFrame({"AU": ["AU5R", "AU4+1"]})
        test_result_df = pd.DataFrame(
            {
                "AU": ["AU5R", "AU4+1"],
                "AU1": [0, 1],
                "AU4": [0, 1],
                "AU5": [1, 0],
            }
        )
        test_df_extracted = extract_action_units(test_df)
        res = np.array(test_df_extracted == test_result_df).mean()
        self.assertEqual(1, res)


if __name__ == "__main__":
    unittest.main()
