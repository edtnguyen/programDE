# Interactive Fig. 3 carrier-seeded neighborhood notebook

Open `fig3_carrierseeded_neighborhood_panels_interactive.ipynb` in Jupyter on Sherlock.

The notebook displays panel f and panel j figures inline after generating them. It uses the carrier-seeded cNMF-neighborhood logic from `ipsc-ec.Hippo.figure.ipynb`:

- target carriers are seed cells
- neighbors are found in z-scored cNMF program-usage space
- neighbors are restricted to the same day
- enrichment compares local target carriers against same-day non-targeting controls excluding the tested target

Before running, check the editable config cell and update `GUIDE_METADATA_TSV` if needed.
