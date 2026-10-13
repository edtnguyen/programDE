# Fig. 3 carrier-seeded UMAP neighborhood panels — previous plotnine style

This package contains an interactive notebook that uses the same carrier-seeded neighborhood enrichment logic as before, but replaces the matplotlib plotting cells with the plotnine style from `ipsc-ec.Hippo.figure.ipynb`.

Default style in the notebook:
- grey all-day background cells
- shared D0-D3 UMAP
- blue-white-red `log2OR` fill scale
- dot size = carrier fraction in seed-cell neighborhood
- black outline = FDR < 0.05
- `theme_bw`, `coord_equal`, grey strip labels, panel borders
- PDF/SVG outputs with editable text

Run on Sherlock cell-by-cell. If `plotnine` is missing, install in your environment:

```bash
pip install plotnine mizani
```

The main output files will be written to `cfg.OUTDIR`:

- `panel_f_carrierseeded_neighborhood_grid_plotnine_previous_style.png/pdf/svg`
- `panel_j_carrierseeded_neighborhood_grid_plotnine_previous_style.png/pdf/svg`
- optional simple color-only versions

The computation still uses:
- target carriers as seed cells
- KNN in z-scored cNMF usage space
- same-day neighborhood restriction
- same-day NT baseline
