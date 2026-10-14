# Phase 33: time evolution of superpositions

Date: 2026-10-14
Roadmap item 2 (brainstorm items 1 and 27): a superposition builder, and the
superposition evolving in time.

## What it shows

Every view so far draws one stationary state, and a stationary state does not
move: |psi|^2 is frozen for all time. The first thing that moves in quantum
mechanics is a superposition of states with different energies. 1s + 2p0 is the
textbook case: the density sloshes up and down the z axis at the Bohr
frequency (E_2 - E_1)/h, which is the classical picture of a radiating dipole.

The new view, "Time evolution", lets you pick up to four hydrogen-like states,
give each a population |c_k|^2 and a phase, and play

    Psi(r, t) = sum_k c_k psi_k(r) exp(-i E_k t / hbar)

on the y = 0 plane.

## Why the y = 0 plane, and why the browser does the clock

On y = 0 every psi_nlm is real in both angular bases (e^{i m phi} = +/-1
there; the Plane view already relies on this). So the whole movie is K real
grids and K energies. The server evaluates the grids once; the browser runs
the phase sum each frame. Nothing is approximated by moving the clock to the
client: the sum is the exact solution of the time-dependent Schrodinger
equation for an isolated atom, given the stationary states.

## Fidelity

EXACT for hydrogen-like systems (H, D, T, muonic H, positronium, He+, generic
Z), with the reduced mass in both psi_k and E_k. The provenance names what
"exact" excludes:

- Non-relativistic: states that share n are degenerate, so 2s + 2p0 does not
  move at all. Fine structure would make them beat, at about alpha^2 times the
  gross frequencies. The view says "stationary" rather than drawing nothing.
- An isolated atom: no spontaneous emission. Real excited components decay
  (2p: 1.6 ns), which is millions of beat periods, so the picture is fine over
  any time it can show, and the caption says so.
- Populations are normalized so that sum |c_k|^2 = 1.

Screened and Hartree-Fock atoms are refused (422). Their orbitals are not the
eigenstates of the many-electron Hamiltonian, so a phase sum over them would
look exact and not be.

## Colour scale

Per-frame normalization would hide the thing the view exists to show (the
density breathing). The scale is fixed for all t at the bound
max over the grid of (sum_k |c_k| |psi_k|)^2, which no frame can exceed. Same
inferno LUT and disclosed gamma as the Plane view.

## API

`POST /api/jobs/superposition` with
`{components: [{n, l, m, weight, phase}], system, basis, resolution}`.
`weight` is a relative population (>= 0, not all zero), `phase` is in radians.
Up to four components, no duplicates. Meta carries the normalized coefficients
(re, im), each E_k as a Quantity, the distinct beat periods in fs, the extent
and the provenance. Data channels `psi0..psi{K-1}`, float32 on the plane
layout.

## URL

One repeated key per state, `sup=n_l_m_weight_phase`, with the phase in units
of pi: `?view=evolve&sup=1_0_0_1_0&sup=2_1_0_1_0`. Underscores because
URLSearchParams leaves them unescaped and a weight carries a `.` and m a `-`.
Absent means the default 1s + 2p0, equal populations, zero phase.

## Not in this phase

Phase colouring of the evolving Psi (roadmap item 4), the 3-D cloud in time,
and an <z>(t) trace (item 5, the expectation panel).
