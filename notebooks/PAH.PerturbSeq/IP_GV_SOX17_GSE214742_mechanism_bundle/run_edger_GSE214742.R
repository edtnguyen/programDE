#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(edgeR)
    library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    stop(
        "Usage: run_edger_GSE214742.R ",
        "<counts.tsv.gz> <metadata.tsv> <output_dir>"
    )
}

counts_path <- normalizePath(args[[1]])
metadata_path <- normalizePath(args[[2]])
output_dir <- normalizePath(args[[3]], mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

counts <- read.delim(
    counts_path,
    row.names = 1,
    check.names = FALSE
)
metadata <- read.delim(
    metadata_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
)

if (!all(metadata$gsm %in% colnames(counts))) {
    missing <- setdiff(metadata$gsm, colnames(counts))
    stop("Counts are missing samples: ", paste(missing, collapse = ", "))
}

counts <- counts[, metadata$gsm, drop = FALSE]
storage.mode(counts) <- "numeric"
counts <- round(counts)

run_contrast <- function(
    contrast_name,
    treatment,
    control,
    batch = NULL
) {
    selected <- metadata$condition %in% c(treatment, control)
    if (!is.null(batch)) {
        selected <- selected & as.character(metadata$experiment_batch) == as.character(batch)
    }

    md <- metadata[selected, , drop = FALSE]
    md$treated <- as.integer(md$condition == treatment)
    md <- md[order(md$pair_id, md$treated), , drop = FALSE]

    pair_table <- table(md$pair_id, md$treated)
    if (
        ncol(pair_table) != 2
        || any(rowSums(pair_table) != 2)
        || any(pair_table[, "0"] != 1)
        || any(pair_table[, "1"] != 1)
    ) {
        stop(
            contrast_name,
            ": paired structure is invalid:\n",
            paste(capture.output(print(pair_table)), collapse = "\n")
        )
    }

    y <- DGEList(counts = counts[, md$gsm, drop = FALSE])
    design <- model.matrix(
        ~ factor(pair_id) + treated,
        data = md
    )
    keep <- filterByExpr(y, design = design)
    y <- y[keep, , keep.lib.sizes = FALSE]
    y <- calcNormFactors(y)
    y <- estimateDisp(y, design, robust = TRUE)
    fit <- glmQLFit(y, design, robust = TRUE)
    test <- glmQLFTest(fit, coef = "treated")

    result <- topTags(test, n = Inf, sort.by = "none")$table
    result$gene <- rownames(result)
    rownames(result) <- NULL
    result$signed_sqrt_F <- sign(result$logFC) * sqrt(pmax(result$F, 0))

    result <- result[
        order(result$PValue),
        c(
            "gene",
            "logFC",
            "logCPM",
            "F",
            "PValue",
            "FDR",
            "signed_sqrt_F"
        )
    ]

    output <- file.path(
        output_dir,
        paste0(contrast_name, "_edgeR.tsv.gz")
    )
    gz <- gzfile(output, "wt")
    write.table(
        result,
        gz,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
    close(gz)

    sox17 <- result[toupper(result$gene) == "SOX17", , drop = FALSE]
    if (nrow(sox17) == 0) {
        sox17 <- data.frame(
            gene = "SOX17",
            logFC = NA_real_,
            logCPM = NA_real_,
            F = NA_real_,
            PValue = NA_real_,
            FDR = NA_real_,
            signed_sqrt_F = NA_real_
        )
    }
    sox17$contrast <- contrast_name
    sox17$treatment <- treatment
    sox17$control <- control
    sox17$n_pairs <- length(unique(md$pair_id))

    cpm_table <- cpm(y, log = TRUE, prior.count = 2)
    cpm_output <- file.path(
        output_dir,
        paste0(contrast_name, "_logCPM.tsv.gz")
    )
    gz_cpm <- gzfile(cpm_output, "wt")
    write.table(
        data.frame(gene = rownames(cpm_table), cpm_table),
        gz_cpm,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
    close(gz_cpm)

    list(
        contrast = contrast_name,
        treatment = treatment,
        control = control,
        batch = if (is.null(batch)) NA_character_ else as.character(batch),
        n_pairs = length(unique(md$pair_id)),
        n_genes_tested = nrow(result),
        result = output,
        SOX17 = sox17
    )
}

contrasts <- list(
    run_contrast(
        "siSOX17_vs_negativeSIRNA",
        "siSOX17",
        "negativeSIRNA"
    ),
    run_contrast(
        "Promoter_Repression_vs_BFP_Repression",
        "Promoter_Repression",
        "BFP_control_Repression"
    ),
    run_contrast(
        "Signal1_Repression_vs_BFP_Repression",
        "Signal1_Repression",
        "BFP_control_Repression",
        batch = "3"
    ),
    run_contrast(
        "Promoter_Activation_vs_BFP_Activation",
        "Promoter_Activation",
        "BFP_control_Activation"
    ),
    run_contrast(
        "siGAPDH_vs_negativeSIRNA",
        "siGAPDH",
        "negativeSIRNA"
    )
)

manifest <- do.call(
    rbind,
    lapply(contrasts, function(x) {
        data.frame(
            contrast = x$contrast,
            treatment = x$treatment,
            control = x$control,
            batch = x$batch,
            n_pairs = x$n_pairs,
            n_genes_tested = x$n_genes_tested,
            result = x$result,
            stringsAsFactors = FALSE
        )
    })
)
write.table(
    manifest,
    file.path(output_dir, "contrast_manifest.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

sox17_qc <- do.call(
    rbind,
    lapply(contrasts, function(x) x$SOX17)
)
write.table(
    sox17_qc,
    file.path(output_dir, "SOX17_contrast_QC.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

# A single normalized logCPM matrix across all samples for program scoring.
all_y <- DGEList(counts = counts)
keep_all <- filterByExpr(
    all_y,
    group = factor(metadata$condition)
)
all_y <- all_y[keep_all, , keep.lib.sizes = FALSE]
all_y <- calcNormFactors(all_y)
all_logcpm <- cpm(all_y, log = TRUE, prior.count = 2)

all_logcpm_output <- file.path(
    output_dir,
    "GSE214742_all_samples_logCPM.tsv.gz"
)
gz_all <- gzfile(all_logcpm_output, "wt")
write.table(
    data.frame(gene = rownames(all_logcpm), all_logcpm),
    gz_all,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)
close(gz_all)

write_json(
    list(
        contrasts = manifest,
        SOX17_QC = sox17_qc,
        all_samples_logCPM = all_logcpm_output
    ),
    file.path(output_dir, "EDGER_COMPLETE.json"),
    pretty = TRUE,
    auto_unbox = TRUE
)

cat("edgeR contrasts completed.\n")
print(sox17_qc)
