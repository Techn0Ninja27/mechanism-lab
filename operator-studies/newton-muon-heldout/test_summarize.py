import unittest
from summarize import curve_stats

def curve(times,losses):
    return [{'step':i*32,'training_seconds':t,'heldout':{'mean_cross_entropy':y}} for i,(t,y) in enumerate(zip(times,losses))]

class CurveTests(unittest.TestCase):
    def test_clipped_area_and_last_observation_are_distinct(self):
        x=curve_stats(curve([0,2,4],[5,3,1]),3)
        self.assertEqual(x['common_time_interpolated'],2)
        self.assertEqual(x['common_time_last_observed'],3)
        self.assertEqual(x['normalized_time_area'],3.5)
    def test_no_extrapolation_or_monotonic_smoothing(self):
        c=curve([0,2,4],[2,1,3])
        self.assertEqual(curve_stats(c,4)['common_time_interpolated'],3)
        with self.assertRaises(ValueError):curve_stats(c,5)
