"""FireFlow: the carried fire streamed as fire does in moving air, rooted at the hand."""
import numpy as np

from cuesync.bending import FireFlow


def flame(h=240, w=160):
    E = np.zeros((h, w, 3), np.float32)
    E[60:200, 60:100] = (1.0, 0.6, 0.2)  # a column of fire, rooted at its bottom row
    return E


def test_its_root_stays_on_the_hand_while_its_top_streams():
    E = flame()
    f = FireFlow(*E.shape[:2], seed=1)
    out = [f(t, E.copy()) for t in range(12)][-1]
    assert np.abs(out[190:200] - E[190:200]).max() < 0.05
    assert np.abs(out[60:90] - E[60:90]).mean() > 0.01


def test_a_moving_hand_leaves_a_cooling_trail_and_a_jump_starts_afresh():
    E, none = flame(), np.zeros_like(flame())
    f = FireFlow(*E.shape[:2], seed=1)
    f(0, E.copy()), f(1, E.copy())
    after = f(2, none.copy())
    assert after[..., 0].max() > 0.2 and after[..., 2].max() < after[..., 0].max()  # red stays longest
    g = FireFlow(*E.shape[:2], seed=1)
    g(0, E.copy()), g(1, E.copy())
    assert g(9, none.copy()).max() == 0.0


def test_a_thrown_fireball_leaves_no_ghost_of_itself_behind():
    E, none = flame(), np.zeros_like(flame())
    f = FireFlow(*E.shape[:2], seed=1, trail=False)
    f(0, E.copy()), f(1, E.copy())
    assert f(2, none.copy()).max() == 0.0


def test_the_same_seed_streams_the_same_fire():
    E = flame()
    a, b = FireFlow(*E.shape[:2], seed=4), FireFlow(*E.shape[:2], seed=4)
    for t in range(6):
        assert np.array_equal(a(t, E.copy(), (1.5, 0.0)), b(t, E.copy(), (1.5, 0.0)))
