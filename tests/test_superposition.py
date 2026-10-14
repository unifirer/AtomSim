import numpy as np
import pytest

from atomsim.analytic.wavefunction import evaluate_state
from atomsim.provenance import Fidelity
from atomsim.superposition import (
    Component,
    evolve_density,
    normalized_coefficients,
    superposition_grid,
)

# 1s + 2p0, equal populations: the textbook sloshing dipole.
SLOSH = [Component(1, 0, 0, 1.0), Component(2, 1, 0, 1.0)]
BEAT = 0.5 - 0.125  # E_2 - E_1 for hydrogen, hartree


def _z_centroid(sg, rho):
    z = sg.axis[:, None]
    return float((rho * z).sum() / rho.sum())


def test_matches_direct_complex_evaluation_at_any_time():
    """The plane grids are real, and the sum over them is the full complex Psi.

    This is the claim the whole design leans on: on y=0 the complex-basis
    psi_nlm has no imaginary part, so the browser's real-grid phase sum loses
    nothing. Checked with m = +/-1, where e^{i m phi} is not trivially 1.
    """
    comps = [Component(2, 1, 1, 0.3, 0.4), Component(3, 2, -1, 0.7, -1.1)]
    sg = superposition_grid(comps, resolution=21)
    t = 7.3
    i, j = 4, 15
    pos = np.array([[sg.axis[j], 0.0, sg.axis[i]]])
    psi = sum(
        c * evaluate_state(k.n, k.l, k.m, pos).values[0] * np.exp(-1j * e.value * t)
        for c, k, e in zip(sg.coefficients, comps, sg.energies)
    )
    assert evolve_density(sg, t)[i, j] == pytest.approx(abs(psi) ** 2, rel=1e-10)


def test_populations_are_normalized_and_phases_kept():
    c = normalized_coefficients([Component(1, 0, 0, 2.0, 0.0), Component(2, 1, 0, 6.0, 1.0)])
    assert np.sum(np.abs(c) ** 2) == pytest.approx(1.0)
    assert abs(c[0]) ** 2 == pytest.approx(0.25)
    assert np.angle(c[1]) == pytest.approx(1.0)


def test_density_returns_after_one_bohr_period():
    sg = superposition_grid(SLOSH, resolution=33)
    period = 2 * np.pi / BEAT
    assert sg.beat_frequencies() == pytest.approx([BEAT])
    rho0 = evolve_density(sg, 0.0)
    np.testing.assert_allclose(evolve_density(sg, period), rho0, rtol=1e-9, atol=1e-14)
    # and it really moved in between
    assert np.abs(evolve_density(sg, period / 2) - rho0).max() > 0.1 * rho0.max()


def test_the_dipole_sloshes_along_z():
    """psi_2p0 > 0 for z > 0, so at t = 0 the in-phase sum piles up at +z and
    half a period later at -z."""
    sg = superposition_grid(SLOSH, resolution=65)
    period = 2 * np.pi / BEAT
    assert _z_centroid(sg, evolve_density(sg, 0.0)) > 0.1
    assert _z_centroid(sg, evolve_density(sg, period / 2)) < -0.1
    assert abs(_z_centroid(sg, evolve_density(sg, period / 4))) < 1e-9


def test_time_average_kills_the_interference():
    """Averaged over a period, cross terms vanish: <rho> = sum_k |c_k|^2 psi_k^2."""
    sg = superposition_grid(SLOSH, resolution=33)
    period = 2 * np.pi / BEAT
    ts = np.arange(64) * period / 64  # exact for a single cosine
    mean = np.mean([evolve_density(sg, t) for t in ts], axis=0)
    incoherent = np.tensordot(np.abs(sg.coefficients) ** 2, sg.psi**2, axes=1)
    np.testing.assert_allclose(mean, incoherent, rtol=1e-9, atol=1e-14)


def test_degenerate_states_are_stationary():
    sg = superposition_grid([Component(2, 0, 0, 1.0), Component(2, 1, 0, 1.0)], resolution=17)
    assert sg.beat_frequencies() == []
    np.testing.assert_allclose(evolve_density(sg, 123.4), evolve_density(sg, 0.0))


def test_zero_weight_component_does_not_beat():
    sg = superposition_grid([Component(1, 0, 0, 1.0), Component(2, 1, 0, 0.0)], resolution=9)
    assert sg.beat_frequencies() == []


def test_reduced_mass_speeds_the_clock():
    """Every E_n scales with mu, so every beat does too: muonic hydrogen at
    mu/m_e ~ 186 beats 186 times faster in a frame 186 times smaller."""
    mu = 185.84
    h = superposition_grid(SLOSH, resolution=9)
    muh = superposition_grid(SLOSH, mu_ratio=mu, resolution=9)
    assert muh.beat_frequencies()[0] == pytest.approx(mu * h.beat_frequencies()[0])
    assert muh.axis[-1] == pytest.approx(h.axis[-1] / mu)


def test_frame_fits_the_largest_n():
    sg = superposition_grid([Component(1, 0, 0, 1.0), Component(3, 1, 0, 1.0)], resolution=9)
    assert sg.axis[-1] == pytest.approx(2.5 * 9)


def test_provenance_is_exact_and_names_what_it_leaves_out():
    sg = superposition_grid(SLOSH, resolution=9)
    assert sg.provenance.fidelity is Fidelity.EXACT
    text = " ".join(sg.provenance.assumptions)
    assert "spontaneous emission" in text
    assert "degenerate" in text


@pytest.mark.parametrize(
    "comps",
    [
        [],
        [Component(1, 0, 0, 1.0)] * 2,
        [Component(n, 0, 0, 1.0) for n in range(1, 6)],
        [Component(1, 0, 0, 0.0)],
        [Component(1, 0, 0, -1.0), Component(2, 0, 0, 2.0)],
        [Component(1, 1, 0, 1.0)],
        [Component(2, 1, 2, 1.0)],
        [Component(1, 0, 0, 1.0, float("nan"))],
    ],
)
def test_bad_mixes_are_refused(comps):
    with pytest.raises(ValueError):
        superposition_grid(comps, resolution=9)
