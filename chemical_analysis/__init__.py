from . import alkalinity, typing
from ._default import WHITEBALANCE_STATS
from ._mobile import AnalysisFacade
from ._model import ContinuousNetwork, EstimationFunction, IntervalNetwork, Network
from ._utils import bgr_to_lab, compute_calibrated_pmf, compute_theoretical_value, correct_predicted_value, correct_theoretical_value, estimate_confidence_in_whitebalance, lab_to_bgr, lab_to_normalized, lab_to_rgb, rgb_to_lab, whitebalance, write_whitebalance_stats

import os
if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from . import sweep
    from ._dataset import ExpandedSampleDataset, ProcessedSampleDataset, SampleDataset, SizedDataset
