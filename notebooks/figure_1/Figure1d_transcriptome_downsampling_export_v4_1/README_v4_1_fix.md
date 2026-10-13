# Figure 1d transcriptome downsampling export v4.1

This is a small fix to v4. The v4 wrapper passed `--config-json`, while the script expected `--config`.

Run on Sherlock:

```bash
unzip Figure1d_transcriptome_downsampling_export_v4_1.zip
cd Figure1d_transcriptome_downsampling_export_v4_1
bash run_fig1d_downsampling_export_on_sherlock_v4_1.sh
```

Output bundle:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_v4/fig1d_transcriptome_downsampling_plot_input_bundle.zip
```

After running, check:

```text
/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/figure1_transcriptome_downsampling_v4/processing_qc.tsv
```

Expected for circular capture: `n_cells_retained` should be close to the BS3 report cell count if called-cell filtering is now matching correctly.
