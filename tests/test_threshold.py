import numpy as np
from person_a_sim.threshold_analysis import crossing, alive_curve, lead_time

def test_crossing_interpolates():
    assert abs(crossing(np.array([0.1, 0.2, 0.3]), np.array([0., 0.4, 0.8]), 0.5) - 0.225) < 1e-9

def test_alive_curve_and_lead():
    ft = np.array([0, 1, 1, 2, 3, -1, -1, -1])   # 8 nodes
    assert np.allclose(alive_curve(ft), [7/8, 5/8, 4/8, 3/8])
    assert lead_time(ft, 0.5) == 3
