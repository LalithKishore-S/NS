import numpy as np
import pytest
from person_a_sim.network_generation import make_graph
from person_a_sim.interdependency import couple, inverse
from person_a_sim.cascade_simulator import simulate, to_csr, giant_mask

N = 1000

@pytest.fixture(scope="module")
def system():
    A, B = to_csr(make_graph("ER", N, 4, seed=1)), to_csr(make_graph("ER", N, 4, seed=2))
    return A, B, couple(N, 1.0, seed=3)

def test_coupling_is_bijection():
    dep = couple(N, 1.0, seed=0)
    assert sorted(dep) == list(range(N))
    assert (inverse(dep)[dep] == np.arange(N)).all()

def test_partial_coupling_fraction():
    assert (couple(N, 0.5, seed=0) >= 0).sum() == 500

@pytest.mark.parametrize("p", [0.1, 0.35, 0.5])
def test_irreversible_and_stable(system, p):
    A, B, dep = system
    r = simulate(A, B, dep, p, seed=0)
    c = r["curve"]
    assert (np.diff(c[:, 1]) <= 0).all() and (np.diff(c[:, 2]) <= 0).all()   # no recovery
    aliveA, aliveB = r["fail_t_A"] < 0, r["fail_t_B"] < 0
    if aliveA.any():   # final state obeys connectivity + dependency rules
        assert (giant_mask(A, aliveA) == aliveA).all()
        assert (giant_mask(B, aliveB) == aliveB).all()
        assert (aliveB[dep[aliveA]]).all() and (aliveA[inverse(dep)[aliveB]]).all()

def test_survival_decreases_with_p(system):
    A, B, dep = system
    s = [np.mean([simulate(A, B, dep, p, seed=k)["S_final"] for k in range(3)]) for p in (0.1, 0.3, 0.5)]
    assert s[0] > s[1] > s[2]
