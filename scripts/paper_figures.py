"""Writes paper/figures/*.pdf and paper/generated/*.tex from paper_data.json.

Figures and tables never call the engine. They read one file, so a figure, a
table and a number quoted in the prose can never disagree. Every number in the
manuscript is a macro from paper/generated/numbers.tex; none is typed by hand.
Run paper_data.py first.
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
    fig.savefig(OUT / f"{name}.svg")  # the KaTeX web version shows these
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
            "no computed result crosses without one", ha="center", fontsize=7,
            style="italic")
    _save(fig, "fig1_architecture")


def radial_compare() -> None:
    """Closed-form u(r) against the finite-difference solver, with the gap."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    for marker, row in zip("os^D", DATA["radial_compare"], strict=True):
        r, ue, un = (np.array(row[k]) for k in ("r", "u_exact", "u_num"))
        name = f"{row['n']}{'spdf'[row['l']]}"
        line, = top.plot(r, ue, lw=1.0, label=name)
        top.plot(r[::12], un[::12], ls="none", marker=marker, ms=2.5, mfc="none",
                 color=line.get_color())
        bottom.plot(r, un - ue, lw=0.9, color=line.get_color())
    top.set(ylabel="$u(r) = rR(r)$")
    top.legend(frameon=False, ncol=2, title="lines exact, symbols numerical",
               title_fontsize=6)
    bottom.set(xlim=(0, 25), xlabel="$r$ (bohr)", ylabel=r"$u_\mathrm{num} - u_\mathrm{exact}$")
    bottom.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    _save(fig, "fig_radial_compare")


def error_budget() -> None:
    """How big each kind of error is, relative to the total energy, against Z."""
    fig, ax = plt.subplots(figsize=ONE_COL)
    eb = DATA["error_budget"]
    ax.semilogy([r["z"] for r in eb], [r["relativity_fraction"] for r in eb],
                color=TIER["approximation"][0], lw=1.1,
                label="neglected relativity (1s, Dirac vs Schrödinger)")
    hf = DATA["hf_totals"]
    z = [r["z"] for r in hf]
    ax.semilogy(z, [r["error_bar_hartree"] / abs(r["energy_hartree"]) for r in hf],
                ls="none", marker="s", ms=4, color=TIER["numerical"][0],
                label="quoted numerical bar / |E|")
    ax.semilogy(z, [r["deviation_hartree"] / abs(r["energy_hartree"]) for r in hf],
                ls="none", marker="o", ms=4, mfc="none", color=TIER["numerical"][0],
                label="actual deviation from Bunge / |E|")
    ax.axvline(9, color="0.6", lw=0.6, ls=":")
    ax.text(9.6, 2.2e-2, "relativity disclosed\nfrom $Z = 9$", fontsize=6, va="top")
    ax.set(xlabel="$Z$", ylabel="relative size", xlim=(0, 37), ylim=(1e-6, 5e-2))
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, "fig_error_budget")


def scf_history() -> None:
    """Self-consistency residual against iteration on the fine mesh."""
    fig, ax = plt.subplots(figsize=ONE_COL)
    for marker, row in zip("osD^v", DATA["hf_totals"], strict=True):
        h = row["residual_history"]
        ax.semilogy(range(1, len(h) + 1), h, marker=marker, ms=3.5, lw=0.9,
                    label=f"{row['symbol']} ({row['coarse_iterations']} coarse + "
                          f"{row['iterations']} fine)")
    ax.set(xlabel="fine-mesh SCF iteration", ylabel="residual (hartree)")
    ax.legend(frameon=False, fontsize=6)
    _save(fig, "fig_scf_history")


def counterfactual_series() -> None:
    """Three counterfactual observables against Z."""
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.0), sharex=True)
    color = TIER["counterfactual"][0]
    ex = DATA["exchange_series"]
    axes[0].plot([r["z"] for r in ex], [-r["exchange_hartree"] for r in ex], marker="D",
                 ms=4, color=color, lw=0.9)
    axes[0].set(ylabel=r"$-(E_\mathrm{HF}-E_\mathrm{H})$ (hartree)")
    ps = DATA["pauli_series"]
    z = [r["z"] for r in ps]
    axes[1].plot(z, [r["radius_ratio"] for r in ps], marker="D", ms=3.5, color=color, lw=0.9)
    axes[1].set(ylabel=r"$\langle r\rangle_\mathrm{off}/\langle r\rangle_\mathrm{on}$")
    for zz in (3, 11):
        axes[1].axvline(zz, color="0.6", lw=0.6, ls=":")
    axes[2].plot(z, [r["collapsed_energy"] - r["variational_energy"] for r in ps],
                 marker="D", ms=3.5, color=color, lw=0.9)
    axes[2].axhline(0, color="0.5", lw=0.6)
    axes[2].set(xlabel="$Z$", ylabel=r"$E_{1s^N} - E_\mathrm{bound}$ (hartree)")
    for ax, tag in zip(axes, "abc", strict=True):
        ax.text(0.93, 0.85, f"({tag})", transform=ax.transAxes, fontsize=7)
    _save(fig, "fig_counterfactual_series")


def zeeman() -> None:
    """Hydrogen n = 2 across the Breit-Rabi crossover."""
    fig, ax = plt.subplots(figsize=ONE_COL)
    z = DATA["zeeman_h2"]
    b = np.array(z["b_tesla"])
    for key, values in z["levels_ghz"].items():
        ls = "-" if key.startswith("l=1") else "--"
        ax.plot(b, values, ls=ls, lw=0.9, color=TIER["approximation"][0])
    ax.set(xlabel="$B$ (tesla)", ylabel="energy from $2p_{3/2}$ at $B=0$ (GHz)", xlim=(0, 2))
    ax.plot([], [], ls="-", color=TIER["approximation"][0], label="$2p$ sublevels")
    ax.plot([], [], ls="--", color=TIER["approximation"][0], label="$2s$ sublevels")
    ax.legend(frameon=False)
    _save(fig, "fig_zeeman")


def curve_of_growth_fig() -> None:
    """Lyman-alpha equivalent width against column, regimes marked."""
    fig, ax = plt.subplots(figsize=ONE_COL)
    c = DATA["curve_of_growth_lya"]
    n, w = np.array(c["column_m2"]), np.array(c["equivalent_width_nm"])
    for regime, marker in (("linear", "o"), ("saturated", "s"), ("damping", "^")):
        mask = np.array([r == regime for r in c["regime"]])
        ax.loglog(n[mask], w[mask], ls="none", marker=marker, ms=3, mfc="none",
                  color=TIER["approximation"][0], label=regime)
    ax.loglog(n, w, lw=0.6, color="0.5", zorder=0)
    ax.set(xlabel="column density in the lower level (m$^{-2}$)",
           ylabel="equivalent width (nm)")
    ax.legend(frameon=False, title=r"Ly$\alpha$, $T=10^4$ K", title_fontsize=6)
    _save(fig, "fig_curve_of_growth")


def sci(x: float, sig: int = 2) -> str:
    """x in math-mode scientific notation at `sig` figures: 1.2\\times10^{-5}."""
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.{sig - 1}e}".split("e")
    exponent = int(exponent)
    return mantissa if exponent == 0 else rf"{mantissa}\times10^{{{exponent}}}"


def tables() -> None:
    """numbers.tex (one macro per quoted number) and the two data tables."""
    out = ROOT / "paper" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    m: dict[str, str] = {}
    meta, census = DATA["meta"], DATA["census"]
    m["Commit"], m["Generated"] = meta["commit"], meta["generated"]
    m["PythonTests"] = str(census["python_tests"])
    m["LibertyCount"] = str(census["frontend_visual_liberty_declarations"])
    for tier, n in census["tier_mentions"].items():
        m["Mentions" + tier.title().replace("_", "")] = str(n)

    bars = DATA["error_bars"]
    m["NErrorBars"] = str(len(bars))
    m["NErrorBarsBracketed"] = str(sum(p["quoted"] > p["actual"] for p in bars))

    def ratios(prefix: str) -> list[float]:
        return [p["quoted"] / p["actual"] for p in bars if p["family"].startswith(prefix)]

    for name, prefix in (("Radial", "radial"), ("HFTotal", "Hartree-Fock total energy vs Bunge"),
                         ("HFOrbital", "Hartree-Fock orbital")):
        r = ratios(prefix)
        m[name + "RatioMin"], m[name + "RatioMax"] = f"{min(r):.2f}", f"{max(r):.2f}"
        m[name + "Count"] = str(len(r))
    m["HydrogenHFRatio"] = f"{ratios('Hartree-Fock total energy vs exact')[0]:.2f}"

    orders = [r["observed_order"] for r in DATA["convergence"] if r["observed_order"]]
    m["ConvOrderMin"], m["ConvOrderMax"] = f"{min(orders):.3f}", f"{max(orders):.3f}"

    nist_rows = DATA["nist_lines"]
    m["NNistLines"] = str(len(nist_rows))
    hydrogenic = [r["relative_error"] for r in nist_rows if r["system"] in ("h", "d")]
    m["HydrogenicRelMax"] = sci(max(hydrogenic))
    m["HydrogenicRelMin"] = sci(min(hydrogenic))
    for key in ("he", "li", "na"):
        rel = sorted(r["relative_error"] for r in nist_rows if r["system"] == key)
        name = key.title()
        m[name + "RelMax"] = f"{100 * rel[-1]:.1f}"
        m[name + "RelMin"] = f"{100 * rel[0]:.1f}"
        m[name + "RelMedian"] = f"{100 * float(np.median(rel)):.1f}"
        m[name + "Within"] = str(sum(x <= 0.05 for x in rel))
        m[name + "Lines"] = str(len(rel))
    worst = max((r for r in nist_rows if r["system"] == "he"), key=lambda r: r["relative_error"])
    m["HeWorstRef"] = f"{worst['reference_nm']:.1f}"
    m["HeWorstComputed"] = f"{worst['computed_nm']:.0f}"

    for row in DATA["hf_totals"]:
        s = row["symbol"]
        m[s + "Deviation"] = sci(row["deviation_hartree"])
        m[s + "Bar"] = sci(row["error_bar_hartree"])
        m[s + "Seconds"] = f"{row['solve_seconds']:.1f}"

    cf = DATA["counterfactuals"]
    m["ExchHe"] = f"{cf['exchange']['He']['exchange_energy_hartree']:.1f}"
    m["ExchBe"] = f"{cf['exchange']['Be']['exchange_energy_hartree']:.4f}"
    m["ExchNe"] = f"{cf['exchange']['Ne']['exchange_energy_hartree']:.3f}"
    ar = cf["pauli_argon"]
    m["ArRealE"] = f"{ar['real_energy_hartree']:.2f}"
    m["ArCollapsedE"] = f"{ar['collapsed_energy_hartree']:.2f}"
    m["ArVariationalE"] = f"{ar['variational_energy_hartree']:.2f}"
    m["ArRealRadius"] = f"{ar['real_radius_bohr']:.3f}"
    m["ArCollapsedRadius"] = f"{ar['collapsed_radius_bohr']:.3f}"
    m["ArRadiusRatio"] = f"{ar['radius_ratio']:.3f}"
    m["ArDisplaced"] = f"{DATA['density_ar']['displaced_charge_electrons']:.3f}"
    diff = np.array(DATA["density_ar"]["hf"]) - np.array(DATA["density_ar"]["gsz"])
    m["ArMaxDiff"] = f"{np.abs(diff).max():.2f}"
    m["ArDensityElectrons"] = f"{DATA['density_pauli_ar']['real']['electrons']:.3f}"

    # Numbers for the expanded text.
    systems = {r["key"]: r for r in DATA["systems"]}
    m["LyaH"] = f"{systems['h']['lyman_alpha_nm']:.4f}"
    m["LyaPs"] = f"{systems['ps']['lyman_alpha_nm']:.2f}"
    m["LyaMuH"] = f"{systems['mu-h']['lyman_alpha_nm']:.4f}"
    m["LyaHeIon"] = f"{systems['he+']['lyman_alpha_nm']:.3f}"
    m["MuMuH"] = f"{systems['mu-h']['mu_ratio']:.3f}"
    relfrac = {r["z"]: r["relativity_fraction"] for r in DATA["error_budget"]}
    for z, name in ((1, "One"), (2, "Two"), (18, "Eighteen"), (36, "ThirtySix")):
        m["RelFrac" + name] = sci(relfrac[z])
    hf = {r["symbol"]: r for r in DATA["hf_totals"]}
    for s in ("He", "Ar"):
        bar = hf[s]["error_bar_hartree"] / abs(hf[s]["energy_hartree"])
        m["RelOverBar" + s] = f"{relfrac[hf[s]['z']] / bar:.0f}"
    radial = DATA["radial_compare"]
    m["RadialDiffMax"] = sci(max(r["max_abs_diff"] for r in radial))
    m["RadialDiffMin"] = sci(min(r["max_abs_diff"] for r in radial))
    m["FineItersMin"] = str(min(r["iterations"] for r in DATA["hf_totals"]))
    m["FineItersMax"] = str(max(r["iterations"] for r in DATA["hf_totals"]))
    m["CoarseItersMin"] = str(min(r["coarse_iterations"] for r in DATA["hf_totals"]))
    m["CoarseItersMax"] = str(max(r["coarse_iterations"] for r in DATA["hf_totals"]))
    m["FinalResidualMax"] = sci(max(r["residual_history"][-1] for r in DATA["hf_totals"]))
    ex = {r["symbol"]: r for r in DATA["exchange_series"]}
    m["ExchMg"] = f"{ex['Mg']['exchange_hartree']:.3f}"
    m["ExchAr"] = f"{ex['Ar']['exchange_hartree']:.3f}"
    ps = {r["z"]: r for r in DATA["pauli_series"]}
    m["PauliRatioNe"] = f"{ps[10]['radius_ratio']:.3f}"
    m["PauliRatioNa"] = f"{ps[11]['radius_ratio']:.3f}"
    m["PauliRatioLi"] = f"{ps[3]['radius_ratio']:.3f}"
    m["PauliRatioBe"] = f"{ps[4]['radius_ratio']:.3f}"
    m["HeBound"] = f"{ps[2]['variational_energy']:.4f}"
    m["HeCollapsed"] = f"{ps[2]['collapsed_energy']:.4f}"
    m["NPauliAtoms"] = str(len(ps))
    zee = DATA["zeeman_h2"]["levels_ghz"]
    zero = [v[0] for v in zee.values()]
    m["FineSplitGHz"] = f"{max(zero) - min(zero):.2f}"
    cog = DATA["curve_of_growth_lya"]
    m["LyaF"] = f"{cog['f']:.4f}"
    m["LyaA"] = sci(cog["einstein_a"], 4)
    m["LyaSigma"] = sci(cog["sigma_nm"])
    m["LyaGamma"] = sci(cog["gamma_nm"])
    m["LyaDampingA"] = sci(cog["gamma_nm"] / (cog["sigma_nm"] * np.sqrt(2)))
    m["NAllowlist"] = str(len(DATA["allowlist"]))
    m["NumpyVersion"], m["ScipyVersion"] = meta["numpy"], meta["scipy"]
    m["PythonVersion"] = meta["python"]

    (out / "numbers.tex").write_text(
        "% Generated by scripts/paper_figures.py from paper/data/paper_data.json. Do not edit.\n"
        + "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in m.items()),
        encoding="utf-8",
    )

    hf = [r"\begin{tabular}{lrcccc}", r"\hline\hline",
          r"Atom & $Z$ & $E_\mathrm{HF}$ (hartree) & Quoted bar & $|E_\mathrm{HF}-E_\mathrm{ref}|$"
          r" & Ratio \\", r"\hline"]
    for row in DATA["hf_totals"]:
        hf.append(rf"{row['symbol']} & {row['z']} & ${row['energy_hartree']:.6f}$ & "
                  rf"${sci(row['error_bar_hartree'])}$ & ${sci(row['deviation_hartree'])}$ & "
                  rf"{row['bar_over_deviation']:.2f} \\")
    hf += [r"\hline\hline", r"\end{tabular}"]
    (out / "table_hf.tex").write_text("\n".join(hf) + "\n", encoding="utf-8")

    names = {"h": "H", "d": "D", "he": "He", "li": "Li", "na": "Na"}
    nist_tab = [r"\begin{tabular}{llcccc}", r"\hline\hline",
                r"System & Model (tier) & Lines & Median & Worst & Reference medium \\",
                r"\hline"]
    for key in ("h", "d", "he", "li", "na"):
        rows = [r for r in nist_rows if r["system"] == key]
        rel = [r["relative_error"] for r in rows]
        nist_tab.append(
            rf"{names[key]} & {rows[0]['model']} ({rows[0]['fidelity']}) & {len(rows)} & "
            rf"${sci(float(np.median(rel)))}$ & ${sci(max(rel))}$ & "
            rf"{rows[0]['reference_medium']} \\")
    nist_tab += [r"\hline\hline", r"\end{tabular}"]
    (out / "table_nist.tex").write_text("\n".join(nist_tab) + "\n", encoding="utf-8")

    def write(name: str, spec: str, header: str, rows: list[str]) -> None:
        body = [rf"\begin{{tabular}}{{{spec}}}", r"\hline\hline", header + r" \\", r"\hline",
                *[row + r" \\" for row in rows], r"\hline\hline", r"\end{tabular}"]
        (out / f"{name}.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    write("table_systems", "lrccc",
          r"System & $Z$ & $\mu/m_e$ & $E_1$ (eV) & Ly$\alpha$ (nm)",
          [rf"{r['name']} & {r['z']} & ${r['mu_ratio']:.8g}$ & ${r['ground_ev']:.4f}$ & "
           rf"${r['lyman_alpha_nm']:.4f}$" for r in DATA["systems"]])
    write("table_allowlist", "lp{0.55\\textwidth}",
          r"Model.field & Why it carries no provenance",
          [(rf"\texttt{{{r['model']}.{r['field']}}} & {r['reason']}").replace("_", r"\_")
           for r in DATA["allowlist"]])
    write("table_pauli", "rcccc",
          r"$Z$ & $E$ (exclusion on) & $E_{1s^N}$ (off) & bound & $\langle r\rangle$ ratio",
          [rf"{r['z']} & ${r['real_energy']:.3f}$ & ${r['collapsed_energy']:.3f}$ & "
           rf"${r['variational_energy']:.3f}$ & {r['radius_ratio']:.3f}"
           for r in DATA["pauli_series"]])
    write("table_budget", "lcccc",
          r"Atom & bar$/|E|$ & deviation$/|E|$ & relativity & relativity / bar",
          [rf"{r['symbol']} & ${sci(r['error_bar_hartree'] / abs(r['energy_hartree']))}$ & "
           rf"${sci(r['deviation_hartree'] / abs(r['energy_hartree']))}$ & "
           rf"${sci(relfrac[r['z']])}$ & "
           rf"{relfrac[r['z']] / (r['error_bar_hartree'] / abs(r['energy_hartree'])):.0f}"
           for r in DATA["hf_totals"]])
    write("table_exchange", "lrc", r"Atom & $Z$ & $E_\mathrm{HF}-E_\mathrm{Hartree}$ (hartree)",
          [rf"{r['symbol']} & {r['z']} & ${r['exchange_hartree']:.4f}$"
           for r in DATA["exchange_series"]])
    write("table_nist_full", "llcccc",
          r"System & Transition & Computed (nm) & NIST (nm) & Relative residual & Medium",
          [rf"{names[r['system']]} & {r['transition']} & ${r['computed_nm']:.4f}$ & "
           rf"${r['reference_nm']:.4f}$ & ${sci(r['relative_error'])}$ & {r['reference_medium']}"
           for r in nist_rows])
    print(f"wrote {len(m)} macros and 8 tables")


def main() -> None:
    tables()
    radial_compare()
    error_budget()
    scf_history()
    counterfactual_series()
    zeeman()
    curve_of_growth_fig()
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
