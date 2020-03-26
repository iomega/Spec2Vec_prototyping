# test functions

import numpy as np
import unittest

from MS_functions import process_peaks, exponential_peak_filter

class ModelGenerationSuite(unittest.TestCase):
    """Basic test cases."""
    
    def test_process_peaks(self):
        """ Basic test of process_peaks function."""
        peak_mass = [100, 150, 200, 300, 500, 510, 1100]
        peak_intensity = [700, 200, 100, 1000, 200, 5, 500]
        peaks = list(zip(peak_mass, peak_intensity))
        
        # Test peak processing function using spectrum with known outcome
        peaks_processed = process_peaks(peaks, min_frag=0, max_frag=1000, 
                          min_intensity_perc=1,
                          exp_intensity_filter=None,
                          min_peaks=0)
        assert peaks_processed == [(200, 100), (150, 200), (500, 200), (100, 700), (300, 1000)], 'expected different peaks or differnt sorting'

    def test_exp_intensity_filter(self):
        """ Basic test of process_peaks function."""
        peak_mass = [100, 150, 200, 300, 500, 510, 1100] + [300] * 100 + [400] * 200
        peak_intensity = [700, 200, 100, 1000, 200, 5, 500] + [4] * 100 + [2] * 200
        peaks = list(zip(peak_mass, peak_intensity))
        
        # Test peak processing function using spectrum with known outcome
        peaks_processed = exponential_peak_filter(np.array(peaks), 0.5, 5, 10)
        assert peaks_processed.shape[0] == 6, 'expected different number of peaks after filtering'

        peaks_processed = exponential_peak_filter(np.array(peaks), 0.2, 5, 10)
        assert peaks_processed.shape[0] == 107, 'expected different number of peaks after filtering'


if __name__ == '__main__':
    unittest.main()