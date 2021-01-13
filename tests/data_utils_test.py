import os
import unittest
import tempfile
import shutil

from market_sentiment.data_utils import *


class TestDataUtils(unittest.TestCase):
    def setUp(self): 
        self.dataset_file = tempfile.mktemp()
        self.manifest_dir = tempfile.mktemp()
        self.sample_data_text = sample_data = \
            """With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability .@positive
According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales .@positive"""
        self.ls_format = \
            """{"data": {"text": "With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability ."}, "predictions": [{"result": [{"value": {"choices": ["Positive"]}, "from_name": "sentiment", "to_name": "text", "type": "choices"}], "score": 1.0}]}
{"data": {"text": "According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales ."}, "predictions": [{"result": [{"value": {"choices": ["Positive"]}, "from_name": "sentiment", "to_name": "text", "type": "choices"}], "score": 1.0}]}"""

        with open(self.dataset_file, 'w') as f:
            f.write(self.sample_data_text)


    def tearDown(self):
        os.remove(self.dataset_file)
        return super().tearDown()

    def test_df_ls(self):
        assert os.path.exists(self.dataset_file)
        
        # Test dataframe to Label Studio format
        df = load_finphrase(self.dataset_file)
        assert df_to_ls(df) == self.ls_format
        