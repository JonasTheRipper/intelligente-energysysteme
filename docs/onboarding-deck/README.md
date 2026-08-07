# SoCal Wildfires — Onboarding Deck

A 27-slide Beamer deck introducing this testbed to new project members
(graduate students with basic palaestrAI familiarity). It covers background and
purpose, the constrained-mutation conceptual model, the technical architecture,
the published results, how to run the scripts, and open ARL research questions.

## Build

```bash
cd docs/onboarding-deck
xelatex socal-wildfires.tex   # run twice, for the total frame count in the footer
```

XeLaTeX is required: the Cal State LA theme requests Montserrat and Open Sans.
Under pdfLaTeX it falls back to Latin Modern and still compiles.

The compiled `socal-wildfires.pdf` is committed so the deck can be read without
a LaTeX toolchain.

## Figures

`\graphicspath` resolves figures from two places:

- `../../analysis/` — the committed result plots (`fire_perimeter_day5.png`,
  `fire_growth.png`, `grid_impact.png`, `five_day_combined.png`,
  `grid_metrics_v04.png`). These are **not** duplicated here, so the deck always
  shows the current committed figures. Regenerating them changes the deck.
- `figures/` — crops of published figures that have no in-repo equivalent:
  - `guardian_architecture.png` — Fig. 2 of the GUARDIAN paper
  - `esm_fig1_calibration.png` — Fig. 1 of "When Wildfire Breaks the Grid"

The remaining diagrams (simulation loop, two-environment agent layout, v0.3→v0.7
release timeline, teacher→CQL→SAC flow) are drawn in TikZ inside the `.tex`.

## Theme

`beamer*CalStateLA.sty` are the Cal State LA theme files, committed unmodified.
The deck overrides the theme's `frametitle` and `footline` templates in its own
preamble — the shipped templates size their `beamercolorbox` as
`wd=\paperwidth` plus `leftskip`/`rightskip`, and beamercolorbox adds the skips
to `wd`, which makes every frame overfull by 63.7pt. The override keeps the
`.sty` files untouched.

## Sources

- Sultan, Logemann & Veith, *GUARDIAN: Geospatial Unseen-event Adversarial
  Reinforcement for Defense and Infrastructure Adaptation*
- Veith & Sultan, *When Wildfire Breaks the Grid: A Validated Extreme-Event
  Testbed for Infrastructure Resilience*
- `README.md`, `CHANGELOG.md`, `docs/AGENTS.md`, `docs/CMA_AGENT.md`
