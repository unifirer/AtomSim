"""Writes paper/figures/*.pdf from paper/data/paper_data.json, and nothing else.

Figures never call the engine. They read the same file the tables do, so a
figure and a table can never disagree about a number. Run paper_data.py first.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "paper" / "data" / "paper_data.json").read_text(encoding="utf-8"))
OUT = ROOT / "paper" / "figures"

# The app's badge hues (web/src/components/Badge.tsx), darkened for white paper:
# the screen values are tuned for a dark background and wash out in print. Every
# tier also gets its own marker, so no figure depends on telling colors apart.
TIER = {
    "exact": ("#0f9e6e", "o"),
    "numerical": ("#2f7fd1", "s"),
    "approximation": ("#b77d00", "^"),
    "counterfactual": ("#d63d8c", "D"),
    "visual_liberty": ("#8a5cb8", "v"),
}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})
ONE_COL = (3.4, 2.6)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {name}.pdf")


def error_bars() -> None:
    """F6: every quoted error bar against the deviation it was supposed to bound."""
    fig, ax = plt.subplots(figsize=ONE_COL)
    groups = {}
    for p in DATA["error_bars"]:
        tier = "numerical" if p["family"].startswith("radial") else "approximation"
        key = "radial solver (grid halving)" if tier == "numerical" else p["family"]
        groups.setdefault((key, tier), []).append((p["quoted"], p["actual"]))
    styles = {"Hartree-Fock total energy vs Bunge 1993": "^",
              "Hartree-Fock orbital energy vs Bunge 1993": "v",
              "Hartree-Fock total energy vs exact": "P"}
    for (key, tier), pts in groups.items():
        q, a = np.array(pts).T
        ax.loglog(q, a, ls="none", marker=styles.get(key, TIER[tier][1]), ms=4,
                  color=TIER[tier][0], mfc="none" if tier == "numerical" else TIER[tier][0],
                  label=f"{key} ({len(pts)})")
    lo = min(min(p["quoted"], p["actual"]) for p in DATA["error_bars"]) / 3
    hi = max(max(p["quoted"], p["actual"]) for p in DATA["error_bars"]) * 3
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.8, label="actual = quoted")
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color="0.85", lw=0, zorder=0)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="quoted error bar (hartree)",
           ylabel="actual deviation (hartree)")
    # The shaded half is where a bar would be smaller than the error it claims
    # to bound. Nothing lands there, so the legend can.
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    _save(fig, "fig6_error_bars")


def convergence() -> None:
    """F3: error against grid step, with the O(h^2) slope the scheme promises."""
    fig, ax = plt.subplots(figsize=ONE_COL)
    cases = {}
    for r in DATA["convergence"]:
        cases.setdefault(r["case"], []).append((r["h_bohr"], r["error_hartree"]))
    for marker, (case, pts) in zip("osD", cases.items(), strict=True):
        h, e = np.array(pts).T
        ax.loglog(h, e, marker=marker, ms=4, color=TIER["numerical"][0], lw=0.8, label=case)
    h = np.array([2e-3, 4e-2])
    ax.loglog(h, 0.05 * h**2, ls="--", color="0.4", lw=0.8, label="slope 2")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="grid step $h$ (bohr)", ylabel=r"$|E - E_\mathrm{exact}|$ (hartree)")
    ax.legend(frameon=False)
    _save(fig, "fig3_convergence")


def nist() -> None:
    """F4: relative residual against NIST for every vendored line, by system."""
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    order = ["h", "d", "he", "li", "na"]
    names = {"h": "H", "d": "D", "he": "He", "li": "Li", "na": "Na"}
    rng = np.random.default_rng(0)  # horizontal jitter only, so points do not overlap
    for i, key in enumerate(order):
        rows = [r for r in DATA["nist_lines"] if r["system"] == key]
        color, marker = TIER[rows[0]["fidelity"]]
        x = i + rng.uniform(-0.18, 0.18, len(rows))
        ax.semilogy(x, [r["relative_error"] for r in rows], ls="none", marker=marker,
                    ms=4, color=color, mfc="none")
    ax.axhline(0.05, color=TIER["approximation"][0], lw=0.8, ls="--")
    ax.text(-0.45, 0.06, "GSZ 5% bar", fontsize=6.5)
    ax.axhline(3e-5, color=TIER["exact"][0], lw=0.8, ls="--")
    ax.text(1.55, 3.6e-5, r"gross-structure bar $3\times10^{-5}$", fontsize=6.5)
    ax.set_xticks(range(len(order)), [names[k] for k in order])
    ax.set(ylabel=r"$|\lambda - \lambda_\mathrm{NIST}|\,/\,\lambda_\mathrm{NIST}$",
           ylim=(2e-6, 1.0))
    _save(fig, "fig4_nist")


def density(name: str, curves: list[tuple[str, list, list, str]], r_max: float) -> None:
    fig, ax = plt.subplots(figsize=ONE_COL)
    for label, grid, values, tier in curves:
        color, _ = TIER[tier]
        ax.plot(grid, values, color=color, lw=1.1, label=label)
    ax.set(xlim=(0, r_max), ylim=(0, None), xlabel="$r$ (bohr)",
           ylabel="$D(r)$ (electrons / bohr)")
    ax.legend(frameon=False)
    _save(fig, name)


def density_compare_argon() -> None:
    """F5: GSZ and HF argon densities. They agree closely enough to hide each
    other, so the lower panel shows the difference the overlay cannot."""
    ar = DATA["density_ar"]
    r, gsz, hf = (np.array(ar[k]) for k in ("grid_bohr", "gsz", "hf"))
    color = TIER[ar["hf_fidelity"]][0]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    top.plot(r, hf, color=color, lw=1.1, label="Hartree-Fock")
    top.plot(r, gsz, color="0.15", lw=0.9, ls=(0, (3, 2)), label="GSZ screened")
    top.set(ylim=(0, None), ylabel="$D(r)$ (electrons / bohr)")
    top.legend(frameon=False)
    bottom.plot(r, hf - gsz, color=color, lw=1.0)
    bottom.axhline(0, color="0.5", lw=0.6)
    bottom.set(xlim=(0, 4.0), xlabel="$r$ (bohr)",
               ylabel=r"$D_\mathrm{HF} - D_\mathrm{GSZ}$")
    _save(fig, "fig5_density_ar")


def architecture() -> None:
    """F1: one value's journey, with the tier it carries at every boundary."""
    fig, ax = plt.subplots(figsize=(7.0, 1.7))
    ax.set(xlim=(0, 10), ylim=(0, 2.2))
    ax.axis("off")
    stops = [
        (0.15, "solver", "Quantity(value, unit,\nProvenance(fidelity,\n"
                         "method, assumptions,\nerror_estimate))"),
        (2.65, "server schema", "Pydantic model\nwith a required\nprovenance field"),
        (5.15, "wire", "JSON meta +\nfloat32 arrays,\ntier in the meta"),
        (7.65, "browser", "Badge drawn next\nto the picture it\ndescribes"),
    ]
    for x, title, body in stops:
        ax.add_patch(FancyBboxPatch((x, 0.25), 2.2, 1.6, boxstyle="square,pad=0",
                                    fc="white", ec="0.2", lw=0.8))
        ax.text(x + 1.1, 1.65, title, ha="center", va="center", weight="bold")
        ax.text(x + 1.1, 0.95, body, ha="center", va="center", fontsize=6.5)
    for x in (2.35, 4.85, 7.35):
        ax.add_patch(FancyArrowPatch((x, 1.05), (x + 0.3, 1.05), arrowstyle="-|>",
                                     mutation_scale=8, color="0.2"))
    ax.text(5.0, 2.1, "the fidelity tier is a required field at every boundary; "
            "a bare float cannot cross", ha="center", fontsize=7, style="italic")
    _save(fig, "fig1_architecture")


def main() -> None:
    architecture()
    convergence()
    nist()
    density_compare_argon()
    error_bars()
    pauli = DATA["density_pauli_ar"]
    density("fig7_pauli_ar", [
        ("with exclusion", pauli["real"]["grid_bohr"], pauli["real"]["values"],
         pauli["real"]["fidelity"]),
        ("exclusion off (1s$^{18}$)", pauli["collapsed"]["grid_bohr"],
         pauli["collapsed"]["values"], pauli["collapsed"]["fidelity"]),
    ], r_max=2.5)


if __name__ == "__main__":
    main()
