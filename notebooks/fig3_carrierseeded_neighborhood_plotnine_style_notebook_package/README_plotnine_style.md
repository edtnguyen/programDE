# Fig. 3 carrier-seeded neighborhood notebook — plotnine style

This is the interactive notebook updated to keep the same UMAP plotting style as `ipsc-ec.Hippo.figure.ipynb`.

Use:

```
fig3_carrierseeded_neighborhood_panels_interactive_plotnine_style.ipynb
```

Key plotting style choices:

- plotnine/ggplot style
- grey D0-D3 background cells
- carrier-seeded neighborhoods colored by local enrichment log2OR
- diverging scale `#053061` / `#F7F7F7` / `#67001F`
- common UMAP limits, `coord_equal`
- Illustrator-editable PDF/SVG with text preserved

In the config cell, set:

```python
cfg.PLOT_STYLE = "simple_color"   # default, matching the simplified previous plots
# or
cfg.PLOT_STYLE = "size_fdr"       # older style with carrier fraction size and FDR outline
```
