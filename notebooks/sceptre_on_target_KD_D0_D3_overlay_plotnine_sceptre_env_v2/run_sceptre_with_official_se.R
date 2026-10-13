
private_lib <- Sys.getenv("SCEPTRE_PRIVATE_R_LIB")
if (!nzchar(private_lib)) stop("SCEPTRE_PRIVATE_R_LIB is empty.")
.libPaths(unique(c(private_lib, .libPaths())))

suppressPackageStartupMessages({
  library(sceptre)
  library(Matrix)
})

parse_bool <- function(x) tolower(trimws(x)) %in% c("true", "t", "1", "yes", "y")
base_dir <- Sys.getenv("SCEPTRE_BASEDIR")
days <- strsplit(Sys.getenv("SCEPTRE_DAYS"), ",", fixed = TRUE)[[1]]
result_filename <- Sys.getenv("SCEPTRE_RESULT_FILENAME")
force_rerun <- parse_bool(Sys.getenv("SCEPTRE_FORCE_RERUN"))
run_calibration <- parse_bool(Sys.getenv("SCEPTRE_RUN_CALIBRATION"))
save_rds <- parse_bool(Sys.getenv("SCEPTRE_SAVE_RDS"))
run_parallel <- parse_bool(Sys.getenv("SCEPTRE_PARALLEL"))
q_threshold <- as.numeric(Sys.getenv("SCEPTRE_Q_THRESHOLD"))
base_seed <- as.integer(Sys.getenv("SCEPTRE_RANDOM_SEED"))
required_version <- package_version(Sys.getenv("SCEPTRE_REQUIRED_VERSION"))
installed_version <- packageVersion("sceptre")
installed_path <- find.package("sceptre")

cat("SCEPTRE version:", as.character(installed_version), "\n")
cat("SCEPTRE package path:", installed_path, "\n")
cat("R library paths:", paste(.libPaths(), collapse = " | "), "\n")
if (installed_version != required_version) {
  stop(paste0("Expected exact SCEPTRE ", required_version, " but loaded ", installed_version, " from ", installed_path, "."))
}
if (!is.finite(q_threshold) || q_threshold <= 0 || q_threshold >= 1) {
  stop("SCEPTRE_Q_THRESHOLD must lie strictly between 0 and 1.")
}

required_result_columns <- c(
  "response_id", "grna_target", "n_nonzero_trt", "n_nonzero_cntrl",
  "pass_qc", "p_value", "fold_change", "se_fold_change",
  "log_2_fold_change", "significant"
)

to_logical_strict <- function(x, column_name) {
  if (is.logical(x)) return(x)
  values <- tolower(trimws(as.character(x)))
  out <- rep(NA, length(values))
  out[values %in% c("true", "t", "1")] <- TRUE
  out[values %in% c("false", "f", "0")] <- FALSE
  unknown <- is.na(out) & !is.na(x)
  if (any(unknown)) {
    stop(paste("Cannot parse", column_name, "as logical; examples:", paste(head(unique(values[unknown])), collapse = ", ")))
  }
  out
}

normalise_result <- function(result, day_label) {
  result <- as.data.frame(result, stringsAsFactors = FALSE)
  missing <- setdiff(required_result_columns, names(result))
  if (length(missing) > 0) {
    stop(paste(day_label, "result lacks required columns:", paste(missing, collapse = ", ")))
  }
  numeric_columns <- c(
    "n_nonzero_trt", "n_nonzero_cntrl", "p_value",
    "fold_change", "se_fold_change", "log_2_fold_change"
  )
  for (column in numeric_columns) result[[column]] <- as.numeric(result[[column]])
  result$pass_qc <- to_logical_strict(result$pass_qc, "pass_qc")
  result$significant <- to_logical_strict(result$significant, "significant")
  result$q_bh <- NA_real_
  eligible <- which(result$pass_qc & is.finite(result$p_value))
  if (length(eligible) == 0) stop(paste(day_label, "has no QC-passing finite p-values."))
  result$q_bh[eligible] <- p.adjust(result$p_value[eligible], method = "BH")
  result$q_significant <- !is.na(result$q_bh) & result$q_bh <= q_threshold
  result$day <- day_label

  finite_fc <- !is.na(result$pass_qc) & result$pass_qc &
    is.finite(result$fold_change) & result$fold_change > 0 &
    is.finite(result$log_2_fold_change)
  if (any(finite_fc, na.rm = TRUE)) {
    discrepancy <- max(abs(log2(result$fold_change[finite_fc]) - result$log_2_fold_change[finite_fc]))
    if (!is.finite(discrepancy) || discrepancy > 1e-6) {
      stop(paste(day_label, "fold_change and log_2_fold_change are inconsistent; max discrepancy =", discrepancy))
    }
  }
  result
}

read_names <- function(path) {
  as.character(read.csv(path, header = FALSE, check.names = FALSE, stringsAsFactors = FALSE)[, 1])
}

for (day_index in seq_along(days)) {
  day <- days[[day_index]]
  day_dir <- file.path(base_dir, paste0(day, "_sceptre_transfer"))
  updated_path <- file.path(day_dir, result_filename)
  historical_path <- file.path(day_dir, "TargetKD.csv")
  result <- NULL
  result_source <- NULL

  if (!force_rerun) {
    for (candidate in c(updated_path, historical_path)) {
      if (!file.exists(candidate)) next
      candidate_result <- tryCatch(
        read.csv(candidate, check.names = FALSE, stringsAsFactors = FALSE),
        error = function(e) NULL
      )
      if (!is.null(candidate_result) && all(required_result_columns %in% names(candidate_result))) {
        result <- normalise_result(candidate_result, day)
        result_source <- candidate
        break
      }
    }
  }

  if (is.null(result)) {
    cat("\n", day, ": running SCEPTRE from transfer inputs.\n", sep = "")
    required_inputs <- c(
      "response_matrix.mtx", "grna_matrix.mtx", "gene_names.csv",
      "guide_names.csv", "cell_names.csv", "covariates.csv",
      "guide_mapping.csv", "gene_symbol_map.csv"
    )
    missing_inputs <- required_inputs[!file.exists(file.path(day_dir, required_inputs))]
    if (length(missing_inputs) > 0) {
      stop(paste(day, "is missing transfer files:", paste(missing_inputs, collapse = ", ")))
    }

    response_mat <- as(readMM(file.path(day_dir, "response_matrix.mtx")), "dgCMatrix")
    grna_mat <- as(readMM(file.path(day_dir, "grna_matrix.mtx")), "dgCMatrix")
    gene_names <- read_names(file.path(day_dir, "gene_names.csv"))
    guide_names <- read_names(file.path(day_dir, "guide_names.csv"))
    cell_names <- read_names(file.path(day_dir, "cell_names.csv"))
    if (length(gene_names) != nrow(response_mat)) stop(paste(day, "gene-name count mismatch."))
    if (length(guide_names) != nrow(grna_mat)) stop(paste(day, "guide-name count mismatch."))
    if (length(cell_names) != ncol(response_mat) || length(cell_names) != ncol(grna_mat)) {
      stop(paste(day, "cell-name count mismatch."))
    }
    rownames(response_mat) <- gene_names
    colnames(response_mat) <- cell_names
    rownames(grna_mat) <- guide_names
    colnames(grna_mat) <- cell_names

    covariates <- read.csv(
      file.path(day_dir, "covariates.csv"), row.names = 1,
      check.names = FALSE, stringsAsFactors = FALSE
    )
    if (!all(cell_names %in% rownames(covariates))) stop(paste(day, "covariates omit cells."))
    covariates <- covariates[cell_names, , drop = FALSE]
    if ("response_p_mito" %in% names(covariates)) {
      names(covariates)[names(covariates) == "response_p_mito"] <- "mt_percent"
    }
    reserved_columns <- c(
      "response_n_nonzero", "response_n_umis", "grna_n_nonzero", "grna_n_umis"
    )
    covariates <- covariates[, !(names(covariates) %in% reserved_columns), drop = FALSE]

    guide_mapping <- read.csv(
      file.path(day_dir, "guide_mapping.csv"),
      check.names = FALSE, stringsAsFactors = FALSE
    )
    gene_map <- read.csv(
      file.path(day_dir, "gene_symbol_map.csv"),
      check.names = FALSE, stringsAsFactors = FALSE
    )

    set.seed(base_seed + day_index - 1L)
    sceptre_object <- import_data(
      response_matrix = response_mat,
      grna_matrix = grna_mat,
      grna_target_data_frame = guide_mapping,
      extra_covariates = covariates,
      moi = "high"
    )

    discovery_pairs <- merge(guide_mapping, gene_map, by = "grna_target")
    discovery_pairs <- unique(discovery_pairs[, c("grna_target", "response_id")])
    if (nrow(discovery_pairs) == 0) stop(paste(day, "has no on-target discovery pairs."))

    analysis_formula <- formula(
      ~ log(response_n_nonzero) + log(response_n_umis) +
        log(grna_n_nonzero) + log(grna_n_umis) + batch
    )
    sceptre_object <- set_analysis_parameters(
      sceptre_object = sceptre_object,
      discovery_pairs = discovery_pairs,
      side = "both",
      grna_integration_strategy = "union",
      formula_object = analysis_formula,
      resampling_approximation = "skew_normal",
      control_group = "complement",
      resampling_mechanism = "crt",
      multiple_testing_method = "BH",
      multiple_testing_alpha = q_threshold
    )
    sceptre_object <- assign_grnas(
      sceptre_object = sceptre_object,
      method = "mixture",
      parallel = run_parallel
    )
    sceptre_object <- run_qc(
      sceptre_object = sceptre_object,
      p_mito_threshold = 0.25
    )

    if (run_calibration) {
      sceptre_object <- run_calibration_check(
        sceptre_object = sceptre_object,
        parallel = run_parallel
      )
      calibration_result <- get_result(
        sceptre_object = sceptre_object,
        analysis = "run_calibration_check"
      )
      write.csv(
        as.data.frame(calibration_result),
        file.path(day_dir, "calibration_result_updated_sceptre.csv"),
        row.names = FALSE
      )
    }

    sceptre_object <- run_discovery_analysis(
      sceptre_object = sceptre_object,
      output_amount = 1,
      parallel = run_parallel
    )
    result <- get_result(
      sceptre_object = sceptre_object,
      analysis = "run_discovery_analysis"
    )
    result <- normalise_result(result, day)
    result_source <- "fresh_sceptre_rerun"

    if (save_rds) {
      saveRDS(sceptre_object, file.path(day_dir, "sceptre_object_v0.10.3_with_SE.rds"))
    }
    writeLines(
      capture.output(sessionInfo()),
      file.path(day_dir, "sceptre_sessionInfo_v0.10.3.txt")
    )
    writeLines(
      capture.output(print(sceptre_object)),
      file.path(day_dir, "sceptre_analysis_summary_v0.10.3.txt")
    )
    rm(
      response_mat, grna_mat, covariates, guide_mapping, gene_map,
      discovery_pairs, sceptre_object
    )
    gc()
  } else {
    cat(day, ": reusing result with standard errors from ", result_source, "\n", sep = "")
  }

  write.csv(result, updated_path, row.names = FALSE)
  writeLines(
    c(
      paste0("sceptre_version=", as.character(installed_version)),
      paste0("sceptre_path=", installed_path),
      paste0("result_source=", result_source),
      paste0("q_threshold=", q_threshold),
      paste0("n_pairs=", nrow(result)),
      paste0("n_qc_passing=", sum(result$pass_qc, na.rm = TRUE))
    ),
    file.path(day_dir, "TargetKD_with_SE_v0.10.3_metadata.txt")
  )
  cat(day, ": wrote ", updated_path, "\n", sep = "")
}
