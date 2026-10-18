import pytest
from person_a_sim.network_generation import make_graph, avg_degree

@pytest.mark.parametrize("fam", ["ER", "BA", "WS", "SBM"])
def test_matched_size_and_degree(fam):
    g = make_graph(fam, n=1000, k=4.0, seed=1)
    assert g.number_of_nodes() == 1000
    assert abs(avg_degree(g) - 4.0) < 0.3
