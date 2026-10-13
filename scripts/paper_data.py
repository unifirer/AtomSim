"""Writes paper/data/paper_data.json: every number the paper prints.

The paper argues that a code should never quietly lie about a number, so it
cannot carry a single hand-typed one. Every table and figure reads this file,
and this file reads the live engine. Regenerate after any physics change:

    python scripts/paper_data.py
"""

import json
import platform
import re
import subprocess
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import scipy

from atomsim.analytic.hydrogen import energy
from atomsim.atoms import aufbau_configuration
from atomsim.density_compare import compare_total_densities
from atomsim.hf_atom import (
    hf_exchange_energy,
    hf_total_radial_density,
    pauli_collapse,
    solve_hartree_fock,
)
from atomsim.hf_reference import load_hf_reference
from atomsim.numerics.radial_solver import solve_radial, solve_radial_with_error
from atomsim.provenance import Fidelity
from atomsim.screened_atom import solve_screened_atom
from atomsim.spectra import (
    compare_lines,
    load_reference,
    screened_transition_lines,
    transition_lines,
)
from atomsim.systems import get_system

# The SCF loop's inner eigensolver reports when one step stops short of its own
# tolerance. The outer loop has the real gate and raises HFConvergenceError
# rather than return an unconverged atom, so these say nothing about the result.
warnings.filterwarnings("ignore", message="Exited", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "data" / "paper_data.json"

CLOSED_SHELL = [("He", 2), ("Be", 4), ("Ne", 10), ("Mg", 12), ("Ar", 18)]

# Bunge et al. (1993) beryllium orbital energies, the same two numbers
# tests/test_hf_atom.py checks against. Only Be has them vendored.
BE_ORBITALS_BUNGE = {"1s": -4.7326699, "2s": -0.3092695}


def _meta() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    return {
        "commit": commit,
        "generated": date.today().isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _hf_totals() -> tuple[list[dict], list[dict]]:
    """Table 3 and the Hartree-Fock half of the error-bar figure."""
    rows, points = [], []
    for symbol, z in CLOSED_SHELL:
        start = time.perf_counter()
        result = solve_hartree_fock(z, z, aufbau_configuration(z))
        seconds = time.perf_counter() - start
        reference = load_hf_reference(symbol)["total_energy_hartree"]
        got = result.total_energy.value
        bar = result.total_energy.provenance.error_estimate
        deviation = abs(got - reference)
        rows.append({
            "symbol": symbol, "z": z, "energy_hartree": got, "error_bar_hartree": bar,
            "reference_hartree": reference, "deviation_hartree": deviation,
            "bar_over_deviation": bar / deviation, "virial_ratio": result.virial_ratio.value,
            "fidelity": result.total_energy.provenance.fidelity.value,
            "solve_seconds": seconds,
        })
        points.append({
            "family": "Hartree-Fock total energy vs Bunge 1993", "label": f"{symbol} total",
            "quoted": bar, "actual": deviation,
        })
        if symbol == "Be":
            for orbital in result.orbitals:
                name = f"{orbital.n}{'spdf'[orbital.l]}"
                published = BE_ORBITALS_BUNGE[name]
                points.append({
                    "family": "Hartree-Fock orbital energy vs Bunge 1993",
                    "label": f"Be {name}",
                    "quoted": orbital.energy.provenance.error_estimate,
                    "actual": abs(orbital.energy.value - published),
                })

    hydrogen = solve_hartree_fock(1, 1, aufbau_configuration(1))
    points.append({
        "family": "Hartree-Fock total energy vs exact", "label": "H total",
        "quoted": hydrogen.total_energy.provenance.error_estimate,
        "actual": abs(hydrogen.total_energy.value - (-0.5)),
    })
    return rows, points


def _radial_points() -> list[dict]:
    """The numerical solver's grid-halving bars against closed-form answers."""
    points = []
    cases = [
        ("Coulomb Z=1", lambda r: -1.0 / r, lambda n: -0.5 / n**2, 120.0),
        ("Coulomb Z=2", lambda r: -2.0 / r, lambda n: -2.0 / n**2, 60.0),
    ]
    for name, potential, exact_of_n, r_max in cases:
        for l in range(3):
            sol = solve_radial_with_error(potential, l=l, r_max=r_max, n_points=6000, n_states=3)
            for k, q in enumerate(sol.energies):
                n = l + 1 + k
                points.append({
                    "family": f"radial solver vs exact, {name}", "label": f"n={n} l={l}",
                    "quoted": q.provenance.error_estimate, "actual": abs(q.value - exact_of_n(n)),
                })
    # 3-D isotropic oscillator, E = 2k + l + 3/2: a second potential shape with no cusp.
    for l in range(3):
        sol = solve_radial_with_error(lambda r: 0.5 * r**2, l=l, r_max=12.0, n_points=1500)
        for k, q in enumerate(sol.energies):
            points.append({
                "family": "radial solver vs exact, oscillator", "label": f"k={k} l={l}",
                "quoted": q.provenance.error_estimate, "actual": abs(q.value - (2 * k + l + 1.5)),
            })
    return points


def _convergence() -> list[dict]:
    """Figure 3: error against step on the uniform grid, with the observed order."""
    cases = [
        ("3D oscillator l=0", lambda r: 0.5 * r**2, 1.5, 0, 12.0, [600, 1200, 2400, 4800]),
        ("Hydrogen 1s", lambda r: -1.0 / r, energy(1).value, 0, 60.0,
         [3000, 6000, 12000, 24000]),
        ("Hydrogen 2p", lambda r: -1.0 / r, energy(2).value, 1, 100.0,
         [3000, 6000, 12000, 24000]),
    ]
    rows = []
    for name, potential, exact, l, r_max, sizes in cases:
        previous = None
        for n_points in sizes:
            e = solve_radial(potential, l=l, r_max=r_max, n_points=n_points, n_states=1)
            err = abs(e.energies[0].value - exact)
            rows.append({
                "case": name, "n_points": n_points, "h_bohr": r_max / (n_points + 1),
                "error_hartree": err,
                "observed_order": float(np.log2(previous / err)) if previous else None,
            })
            previous = err
    return rows


def _row(system: str, model: str, medium: str, transition: str, line, ref) -> dict:
    computed = line.wavelength.value
    return {
        "system": system, "model": model, "reference_medium": medium,
        "transition": transition, "reference_label": ref.label,
        "computed_nm": computed, "reference_nm": ref.wavelength_nm,
        "delta_nm": computed - ref.wavelength_nm,
        "relative_error": abs(computed - ref.wavelength_nm) / ref.wavelength_nm,
        "fidelity": line.wavelength.provenance.fidelity.value,
    }


def _nist_lines() -> list[dict]:
    """Table 4 and figure 4: computed wavelengths against vendored NIST ASD lines,
    through the same compare_lines the app's Spectrum view uses. Every vendored
    line must come back paired; a dropped one would shrink the table quietly."""
    rows = []
    for key, model, lines in (
        ("h", "Bohr levels, reduced mass", transition_lines(get_system("h"), n_max=6)),
        ("d", "Bohr levels, reduced mass", transition_lines(get_system("d"), n_max=6)),
        *(
            (key, "GSZ screened potential", screened_transition_lines(
                solve_screened_atom(z=z, n_electrons=z, config=aufbau_configuration(z))))
            for key, z in (("he", 2), ("li", 3), ("na", 11))
        ),
    ):
        reference = load_reference(key)
        pairs = compare_lines(lines, reference, tolerance_relative=1.0, window_relative=0.25)
        if len(pairs) != len(reference.lines):
            missing = len(reference.lines) - len(pairs)
            raise LookupError(f"{key}: {missing} reference lines unpaired")
        by_nm = {r.wavelength_nm: r for r in reference.lines}
        for c in pairs:
            ln = c.line
            transition = (f"{ln.n_upper}-{ln.n_lower}" if key in ("h", "d") else
                          f"{ln.n_upper}{'spdf'[ln.l_upper]}-{ln.n_lower}{'spdf'[ln.l_lower]}")
            rows.append(_row(key, model, reference.medium, transition, ln, by_nm[c.reference_nm]))
    return rows


def _counterfactuals() -> dict:
    exchange = {}
    for symbol, z in (("He", 2), ("Be", 4), ("Ne", 10)):
        q = hf_exchange_energy(z, z, aufbau_configuration(z))
        exchange[symbol] = {"exchange_energy_hartree": q.value,
                            "fidelity": q.provenance.fidelity.value}
    ar = pauli_collapse(18)
    return {
        "exchange": exchange,
        "pauli_argon": {
            "real_energy_hartree": ar.real.total_energy.value,
            "collapsed_energy_hartree": ar.collapsed.total_energy.value,
            "binding_change_hartree": ar.binding_change.value,
            "real_radius_bohr": ar.real_radius.value,
            "collapsed_radius_bohr": ar.collapsed_radius.value,
            "radius_ratio": ar.radius_ratio.value,
            "variational_energy_hartree": ar.variational_energy.value,
            "fidelity": ar.collapsed.total_energy.provenance.fidelity.value,
        },
    }


def _density_argon() -> dict:
    cmp_ = compare_total_densities(18, 18)
    return {
        "grid_bohr": cmp_.grid.tolist(),
        "gsz": cmp_.gsz.values.tolist(),
        "hf": cmp_.hf.values.tolist(),
        "displaced_charge_electrons": cmp_.displaced_charge.value,
        "gsz_fidelity": cmp_.gsz.provenance.fidelity.value,
        "hf_fidelity": cmp_.hf.provenance.fidelity.value,
    }


def _census() -> dict:
    """How often each tier is named in the engine source, and the test count.

    A name count, not an assignment count: `is Fidelity.EXACT` in a check counts
    as well as `fidelity=Fidelity.EXACT` in a constructor. The paper says so.
    """
    counts = {f.value: 0 for f in Fidelity}
    for path in (ROOT / "src" / "atomsim").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for f in Fidelity:
            counts[f.value] += text.count(f"Fidelity.{f.name}")
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    tests = int(re.search(r"(\d+) tests? collected", collected).group(1))
    # The engine never assigns VISUAL_LIBERTY: presentation happens in the
    # browser, so that is where the declarations live.
    liberties = sum(
        path.read_text(encoding="utf-8").count('fidelity: "visual_liberty"')
        for pattern in ("*.ts", "*.tsx")
        for path in (ROOT / "web" / "src").rglob(pattern)
        if ".test." not in path.name
    )
    return {"tier_mentions": counts, "python_tests": tests,
            "frontend_visual_liberty_declarations": liberties}


def _density_pauli_argon() -> dict:
    """Figure 7: argon's whole-atom D(r) with exclusion, and collapsed to 1s^18."""
    out = {}
    for name, kwargs in (("real", {}), ("collapsed", {"exchange": False, "pauli": False})):
        d = hf_total_radial_density(18, 18, **kwargs)
        out[name] = {
            "grid_bohr": d.grid.tolist(), "values": d.values.tolist(),
            "electrons": float(np.trapezoid(d.values, d.grid)),
            "fidelity": d.provenance.fidelity.value,
        }
    return out


def main() -> None:
    hf_rows, hf_points = _hf_totals()
    data = {
        "meta": _meta(),
        "census": _census(),
        "hf_totals": hf_rows,
        "error_bars": _radial_points() + hf_points,
        "convergence": _convergence(),
        "nist_lines": _nist_lines(),
        "counterfactuals": _counterfactuals(),
        "density_ar": _density_argon(),
        "density_pauli_ar": _density_pauli_argon(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
