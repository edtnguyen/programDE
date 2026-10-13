#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(
    paste(
      "Usage: run_edger_GSE185479.R",
      "<counts.tsv.gz> <metadata.tsv> <de.tsv.gz>",
      "<logcpm.tsv.gz> <output_dir>"
    )
  )
}

counts_path <- args[[1]]
metadata_path <- args[[2]]
de_path <- args[[3]]
logcpm_path <- args[[4]]
output_dir <- args[[5]]

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

counts_df <- read.delim(
  gzfile(counts_path),
  header = TRUE,
  sep = "\t",
  check.names = FALSE,
  stringsAsFactors = FALSE
)

if (ncol(counts_df) < 3) {
  stop("Pseudobulk count table has too few columns.")
}

gene_names <- counts_df[[1]]
counts <- as.matrix(counts_df[, -1, drop = FALSE])
rownames(counts) <- gene_names
storage.mode(counts) <- "integer"

metadata <- read.delim(
  metadata_path,
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)

required_columns <- c("donor", "condition", "gsm")
missing_columns <- setdiff(required_columns, colnames(metadata))
if (length(missing_columns) > 0) {
  stop(
    paste(
      "Metadata is missing:",
      paste(missing_columns, collapse = ", ")
    )
  )
}

metadata <- metadata[match(colnames(counts), metadata$donor), , drop = FALSE]
if (any(is.na(metadata$donor))) {
  stop("Could not match all pseudobulk columns to donor metadata.")
}

condition <- factor(metadata$condition, levels = c("healthy", "PAH"))
if (any(is.na(condition))) {
  stop("Conditions must be exactly 'healthy' or 'PAH'.")
}
if (!all(table(condition) == c(3, 3))) {
  stop(
    paste(
      "Expected 3 healthy and 3 PAH donors; observed:",
      paste(names(table(condition)), table(condition), collapse = "; ")
    )
  )
}

# Save all-gene TMM logCPM for program scoring.
y_all <- DGEList(counts = counts)
y_all <- calcNormFactors(y_all, method = "TMM")
logcpm <- cpm(y_all, log = TRUE, prior.count = 2)
logcpm_df <- data.frame(
  gene = rownames(logcpm),
  logcpm,
  check.names = FALSE
)
write.table(
  logcpm_df,
  gzfile(logcpm_path),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

design <- model.matrix(~ condition)
keep <- filterByExpr(y_all, design = design)
if (sum(keep) < 1000) {
  stop(
    paste(
      "Only",
      sum(keep),
      "genes passed filterByExpr; expected at least 1000."
    )
  )
}

y <- DGEList(counts = counts[keep, , drop = FALSE])
y <- calcNormFactors(y, method = "TMM")
y <- estimateDisp(y, design, robust = TRUE)
fit <- glmQLFit(y, design, robust = TRUE)
test <- glmQLFTest(fit, coef = "conditionPAH")

de <- topTags(test, n = Inf, sort.by = "none")$table
de$gene <- rownames(de)
de$signed_sqrt_F <- sign(de$logFC) * sqrt(de$F)
de <- de[
  ,
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
de <- de[order(de$PValue), , drop = FALSE]

write.table(
  de,
  gzfile(de_path),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

pdf(file.path(output_dir, "GSE185479_edgeR_MDS.pdf"))
plotMDS(
  y,
  labels = metadata$donor,
  main = "GSE185479 PAEC pseudobulk MDS"
)
dev.off()

write.table(
  data.frame(
    donor = metadata$donor,
    condition = condition,
    library_size = y_all$samples$lib.size,
    norm_factor = y_all$samples$norm.factors
  ),
  file.path(output_dir, "GSE185479_edgeR_library_summary.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

capture.output(
  sessionInfo(),
  file = file.path(output_dir, "R_sessionInfo_edgeR.txt")
)

cat("edgeR analysis completed.\n")
cat("Genes tested:", nrow(de), "\n")
cat("FDR < 0.05:", sum(de$FDR < 0.05), "\n")
