#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(GEOquery)
    library(Biobase)
    library(limma)
    library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    stop(
        "Usage: prepare_GSE70456.R ",
        "<metadata.tsv> <cache_dir> <output_dir>"
    )
}

metadata_path <- normalizePath(args[[1]])
cache_dir <- normalizePath(args[[2]], mustWork = FALSE)
output_dir <- normalizePath(args[[3]], mustWork = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

metadata <- read.delim(
    metadata_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
)

gse_list <- getGEO(
    "GSE70456",
    GSEMatrix = TRUE,
    AnnotGPL = TRUE,
    destdir = cache_dir,
    getGPL = TRUE
)
if (length(gse_list) != 1L) {
    stop("Expected one ExpressionSet for GSE70456.")
}
gse <- gse_list[[1]]

expr <- exprs(gse)
if (max(expr, na.rm = TRUE) > 100) {
    expr <- log2(pmax(expr, 1))
}

sample_ids <- colnames(expr)
missing <- setdiff(metadata$gsm, sample_ids)
if (length(missing)) {
    stop(
        "GSE70456 ExpressionSet is missing: ",
        paste(missing, collapse = ", ")
    )
}
expr <- expr[, metadata$gsm, drop = FALSE]

annotation <- fData(gse)
candidate_columns <- c(
    "Gene Symbol",
    "GENE_SYMBOL",
    "gene_symbol",
    "Symbol",
    "SYMBOL",
    "Gene symbol",
    "gene_assignment"
)
symbol_column <- candidate_columns[
    candidate_columns %in% colnames(annotation)
][1]
if (is.na(symbol_column)) {
    stop(
        "Could not find a gene-symbol column in GPL annotation: ",
        paste(colnames(annotation), collapse = ", ")
    )
}

symbols <- as.character(annotation[[symbol_column]])
if (symbol_column == "gene_assignment") {
    # PrimeView gene_assignment is commonly formatted as:
    # accession // symbol // description // ...
    split_values <- strsplit(symbols, " // ", fixed = TRUE)
    symbols <- vapply(
        split_values,
        function(parts) {
            if (length(parts) >= 2L) parts[[2]] else NA_character_
        },
        character(1)
    )
}
symbols <- toupper(trimws(symbols))
valid <- !is.na(symbols) & nzchar(symbols) & symbols != "---"
expr <- expr[valid, , drop = FALSE]
symbols <- symbols[valid]

# limma::avereps averages probes sharing a symbol.
expr_gene <- avereps(expr, ID = symbols)
expr_gene <- expr_gene[
    apply(expr_gene, 1, function(x) is.finite(var(x)) && var(x) > 0),
    ,
    drop = FALSE
]

expression_path <- file.path(
    output_dir,
    "GSE70456_log2_expression.tsv.gz"
)
connection <- gzfile(expression_path, "wt")
write.table(
    data.frame(gene = rownames(expr_gene), expr_gene),
    connection,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)
close(connection)

metadata <- metadata[match(colnames(expr_gene), metadata$gsm), ]
write.table(
    metadata,
    file.path(output_dir, "GSE70456_metadata.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

run_contrast <- function(contrast_name) {
    md <- metadata[metadata$contrast == contrast_name, , drop = FALSE]
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

    selected <- expr_gene[, md$gsm, drop = FALSE]
    design <- model.matrix(
        ~ factor(pair_id) + treated,
        data = md
    )
    fit <- eBayes(lmFit(selected, design), robust = TRUE)
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
    result$signed_stat <- result$t
    result$method <- "paired_limma_GSE70456"
    result <- result[
        order(result$PValue),
        c(
            "gene",
            "logFC",
            "AveExpr",
            "t",
            "PValue",
            "FDR",
            "signed_stat",
            "method"
        )
    ]

    path <- file.path(
        output_dir,
        paste0(contrast_name, "_limma.tsv.gz")
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

contrast_names <- unique(metadata$contrast)
bmpr2_qc <- do.call(rbind, lapply(contrast_names, run_contrast))
write.table(
    bmpr2_qc,
    file.path(output_dir, "GSE70456_BMPR2_QC.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

write_json(
    list(
        accession = "GSE70456",
        expression = expression_path,
        metadata = file.path(output_dir, "GSE70456_metadata.tsv"),
        contrasts = contrast_names,
        BMPR2_QC = bmpr2_qc,
        symbol_column = symbol_column,
        n_genes = nrow(expr_gene),
        n_samples = ncol(expr_gene)
    ),
    file.path(output_dir, "GSE70456_COMPLETE.json"),
    pretty = TRUE,
    auto_unbox = TRUE
)

cat("GSE70456 preparation and paired limma completed.\n")
print(bmpr2_qc)
