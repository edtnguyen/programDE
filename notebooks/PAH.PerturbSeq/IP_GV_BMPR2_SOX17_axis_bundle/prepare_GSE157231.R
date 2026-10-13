#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(edgeR)
    library(limma)
    library(AnnotationDbi)
    library(org.Hs.eg.db)
    library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    stop(
        "Usage: prepare_GSE157231.R ",
        "<counts.tsv.gz> <metadata.tsv> <output_dir>"
    )
}

counts_path <- normalizePath(args[[1]])
metadata_path <- normalizePath(args[[2]])
output_dir <- normalizePath(args[[3]], mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

metadata <- read.delim(
    metadata_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
)
raw <- read.delim(
    counts_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
)
if (ncol(raw) < 3L) {
    stop("GSE157231 count matrix has fewer than three columns.")
}

gene_values <- as.character(raw[[1]])
counts <- as.matrix(raw[, -1, drop = FALSE])
storage.mode(counts) <- "numeric"
rownames(counts) <- gene_values

normalize_name <- function(x) {
    tolower(gsub("[^a-z0-9]+", "", x))
}

count_names <- colnames(counts)
normalized_count_names <- normalize_name(count_names)

match_column <- function(gsm, title) {
    candidates <- c(
        which(normalized_count_names == normalize_name(gsm)),
        which(normalized_count_names == normalize_name(title))
    )
    candidates <- unique(candidates)
    if (length(candidates) == 0L) {
        # Last resort: title contained within count header or vice versa.
        title_key <- normalize_name(title)
        candidates <- which(
            grepl(title_key, normalized_count_names, fixed = TRUE)
            | grepl(normalized_count_names, title_key, fixed = TRUE)
        )
    }
    if (length(candidates) != 1L) {
        stop(
            "Could not uniquely map ",
            gsm,
            " / ",
            title,
            " to count columns. Count headers:\n",
            paste(count_names, collapse = "\n")
        )
    }
    candidates[[1]]
}

column_indices <- vapply(
    seq_len(nrow(metadata)),
    function(i) {
        match_column(
            metadata$gsm[[i]],
            metadata$sample_title[[i]]
        )
    },
    integer(1)
)
counts <- counts[, column_indices, drop = FALSE]
colnames(counts) <- metadata$gsm

# Convert Ensembl IDs to human gene symbols when needed.
stripped <- sub("\\.[0-9]+$", "", rownames(counts))
ensembl_fraction <- mean(grepl("^ENSG[0-9]+$", stripped))
if (ensembl_fraction > 0.5) {
    symbols <- mapIds(
        org.Hs.eg.db,
        keys = stripped,
        keytype = "ENSEMBL",
        column = "SYMBOL",
        multiVals = "first"
    )
} else {
    symbols <- toupper(stripped)
}
valid <- !is.na(symbols) & nzchar(symbols)
counts <- counts[valid, , drop = FALSE]
symbols <- toupper(as.character(symbols[valid]))

# Sum rows mapping to the same symbol.
counts <- rowsum(counts, group = symbols, reorder = TRUE)
counts <- round(counts)
counts <- counts[rowSums(counts) > 0, , drop = FALSE]

# Aggregate technical replicates so patients are the independent units.
biological_ids <- unique(metadata$biological_sample)
aggregated <- sapply(
    biological_ids,
    function(sample_id) {
        rowSums(
            counts[
                ,
                metadata$biological_sample == sample_id,
                drop = FALSE
            ]
        )
    }
)
colnames(aggregated) <- biological_ids

aggregated_metadata <- metadata[
    match(biological_ids, metadata$biological_sample),
    c(
        "biological_sample",
        "treatment",
        "timepoint",
        "patient",
        "pair_id",
        "treated",
        "contrast"
    )
]
write.table(
    data.frame(gene = rownames(aggregated), aggregated),
    gzfile(
        file.path(
            output_dir,
            "GSE157231_aggregated_counts.tsv.gz"
        ),
        "wt"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)
write.table(
    aggregated_metadata,
    file.path(output_dir, "GSE157231_metadata.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

all_y <- DGEList(counts = aggregated)
keep_all <- filterByExpr(
    all_y,
    group = factor(aggregated_metadata$contrast)
)
all_y <- all_y[keep_all, , keep.lib.sizes = FALSE]
all_y <- calcNormFactors(all_y)
all_logcpm <- cpm(all_y, log = TRUE, prior.count = 2)

expression_path <- file.path(
    output_dir,
    "GSE157231_logCPM.tsv.gz"
)
con <- gzfile(expression_path, "wt")
write.table(
    data.frame(gene = rownames(all_logcpm), all_logcpm),
    con,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)
close(con)

run_contrast <- function(contrast_name) {
    md <- aggregated_metadata[
        aggregated_metadata$contrast == contrast_name,
        ,
        drop = FALSE
    ]
    md$treated <- as.numeric(md$treated)
    md <- md[order(md$pair_id, md$treated), , drop = FALSE]

    pair_table <- table(md$pair_id, md$treated)
    if (
        ncol(pair_table) != 2L
        || any(rowSums(pair_table) != 2L)
        || any(pair_table[, "0"] != 1L)
        || any(pair_table[, "1"] != 1L)
    ) {
        stop(contrast_name, ": invalid paired design.")
    }

    y <- DGEList(
        counts = aggregated[, md$biological_sample, drop = FALSE]
    )
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

    result <- topTags(
        test,
        n = Inf,
        sort.by = "none"
    )$table
    result$gene <- rownames(result)
    rownames(result) <- NULL
    result$signed_stat <- sign(result$logFC) * sqrt(pmax(result$F, 0))
    result$method <- "paired_edgeR_aggregated_technical_replicates"
    result <- result[
        order(result$PValue),
        c(
            "gene",
            "logFC",
            "logCPM",
            "F",
            "PValue",
            "FDR",
            "signed_stat",
            "method"
        )
    ]

    path <- file.path(
        output_dir,
        paste0(contrast_name, "_edgeR.tsv.gz")
    )
    con <- gzfile(path, "wt")
    write.table(
        result,
        con,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
    close(con)

    bmpr2 <- result[toupper(result$gene) == "BMPR2", , drop = FALSE]
    if (nrow(bmpr2) == 0L) {
        bmpr2 <- data.frame(
            gene = "BMPR2",
            logFC = NA_real_,
            PValue = NA_real_,
            FDR = NA_real_
        )
    }
    bmpr2$contrast <- contrast_name
    bmpr2
}

contrast_names <- unique(aggregated_metadata$contrast)
bmpr2_qc <- do.call(rbind, lapply(contrast_names, run_contrast))
write.table(
    bmpr2_qc,
    file.path(output_dir, "GSE157231_BMPR2_QC.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

write_json(
    list(
        accession = "GSE157231",
        counts = counts_path,
        aggregated_counts = file.path(
            output_dir,
            "GSE157231_aggregated_counts.tsv.gz"
        ),
        logCPM = expression_path,
        metadata = file.path(output_dir, "GSE157231_metadata.tsv"),
        contrasts = contrast_names,
        BMPR2_QC = bmpr2_qc,
        technical_replicates_aggregated = TRUE,
        n_genes = nrow(aggregated),
        n_biological_samples = ncol(aggregated)
    ),
    file.path(output_dir, "GSE157231_COMPLETE.json"),
    pretty = TRUE,
    auto_unbox = TRUE
)

cat("GSE157231 preparation and paired edgeR completed.\n")
print(bmpr2_qc)
