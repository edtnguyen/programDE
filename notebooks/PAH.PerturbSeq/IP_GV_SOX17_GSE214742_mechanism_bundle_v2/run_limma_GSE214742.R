#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(limma)
    library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    stop(
        "Usage: run_limma_GSE214742.R ",
        "<log2_abundance.tsv.gz> <metadata.tsv> <output_dir>"
    )
}

expression_path <- normalizePath(args[[1]])
metadata_path <- normalizePath(args[[2]])
output_dir <- normalizePath(args[[3]], mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

expression <- read.delim(
    expression_path,
    row.names = 1,
    check.names = FALSE
)
metadata <- read.delim(
    metadata_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
)

if (!all(metadata$gsm %in% colnames(expression))) {
    missing <- setdiff(metadata$gsm, colnames(expression))
    stop(
        "Expression matrix is missing samples: ",
        paste(missing, collapse = ", ")
    )
}

expression <- as.matrix(expression[, metadata$gsm, drop = FALSE])
storage.mode(expression) <- "numeric"

if (any(!is.finite(expression))) {
    stop("Expression matrix contains NaN or infinite values.")
}
if (any(expression < 0)) {
    stop("Log2 abundance matrix contains negative values.")
}

run_contrast <- function(
    contrast_name,
    treatment,
    control,
    batch = NULL
) {
    selected <- metadata$condition %in% c(treatment, control)
    if (!is.null(batch)) {
        selected <- selected & (
            as.character(metadata$experiment_batch)
            == as.character(batch)
        )
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

    expr <- expression[, md$gsm, drop = FALSE]
    keep <- apply(expr, 1, function(values) {
        is.finite(var(values)) && var(values) > 0
    })
    expr <- expr[keep, , drop = FALSE]

    design <- model.matrix(
        ~ factor(pair_id) + treated,
        data = md
    )

    fit <- lmFit(expr, design)
    fit <- eBayes(fit, robust = TRUE)
    result <- topTable(
        fit,
        coef = "treated",
        number = Inf,
        sort.by = "none"
    )

    result$gene <- rownames(result)
    rownames(result) <- NULL
    result$PValue <- result$P.Value
    result$FDR <- result$adj.P.Val

    # Retain the historical column name expected by the Python GSEA code.
    # For this abundance-only dataset it contains limma's moderated t statistic,
    # not sqrt(F).
    result$signed_sqrt_F <- result$t
    result$method <- "paired_limma_log2_abundance"

    result <- result[
        order(result$PValue),
        c(
            "gene",
            "logFC",
            "AveExpr",
            "t",
            "PValue",
            "FDR",
            "signed_sqrt_F",
            "method"
        )
    ]

    output <- file.path(
        output_dir,
        paste0(contrast_name, "_limma.tsv.gz")
    )
    connection <- gzfile(output, "wt")
    write.table(
        result,
        connection,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
    close(connection)

    sox17 <- result[
        toupper(result$gene) == "SOX17",
        ,
        drop = FALSE
    ]
    if (nrow(sox17) == 0) {
        sox17 <- data.frame(
            gene = "SOX17",
            logFC = NA_real_,
            AveExpr = NA_real_,
            t = NA_real_,
            PValue = NA_real_,
            FDR = NA_real_,
            signed_sqrt_F = NA_real_,
            method = "paired_limma_log2_abundance"
        )
    }
    sox17$contrast <- contrast_name
    sox17$treatment <- treatment
    sox17$control <- control
    sox17$n_pairs <- length(unique(md$pair_id))

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
    lapply(contrasts, function(item) {
        data.frame(
            contrast = item$contrast,
            treatment = item$treatment,
            control = item$control,
            batch = item$batch,
            n_pairs = item$n_pairs,
            n_genes_tested = item$n_genes_tested,
            result = item$result,
            method = "paired_limma_log2_abundance",
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
    lapply(contrasts, function(item) item$SOX17)
)
write.table(
    sox17_qc,
    file.path(output_dir, "SOX17_contrast_QC.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

write_json(
    list(
        method = "paired_limma_log2_abundance",
        expression_matrix = expression_path,
        contrasts = manifest,
        SOX17_QC = sox17_qc
    ),
    file.path(output_dir, "LIMMA_COMPLETE.json"),
    pretty = TRUE,
    auto_unbox = TRUE
)

cat("Paired limma contrasts completed.\n")
print(sox17_qc)
