
parse_bool <- function(x) tolower(trimws(x)) %in% c("true", "t", "1", "yes", "y")
env_or_default <- function(name, default) {
  value <- Sys.getenv(name)
  if (nzchar(value)) value else default
}
private_lib <- env_or_default(
  "SCEPTRE_PRIVATE_R_LIB",
  "/oak/stanford/groups/engreitz/Users/tri/Rlibs/sceptre_0.10.3"
)
required_version_text <- env_or_default("SCEPTRE_REQUIRED_VERSION", "0.10.3")
required_version <- package_version(required_version_text)
repo <- env_or_default("SCEPTRE_GITHUB_REPO", "Katsevich-Lab/sceptre")
ref <- env_or_default("SCEPTRE_GITHUB_REF", "v0.10.3")
install_if_needed <- parse_bool(env_or_default("SCEPTRE_INSTALL_IF_NEEDED", "true"))
cran_repository <- env_or_default("SCEPTRE_CRAN_REPOSITORY", "https://cloud.r-project.org")
output_dir <- env_or_default("SCEPTRE_OUTPUT_DIR", private_lib)

if (!nzchar(private_lib)) stop("SCEPTRE_PRIVATE_R_LIB is empty.")
dir.create(private_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(unique(c(private_lib, .libPaths())))

package_location <- function(package_name) {
  tryCatch(find.package(package_name, quiet = TRUE), error = function(e) "")
}
package_version_from_path <- function(path) {
  if (!nzchar(path)) return(package_version("0.0.0"))
  description_path <- file.path(path, "DESCRIPTION")
  if (!file.exists(description_path)) return(package_version("0.0.0"))
  package_version(read.dcf(description_path, fields = "Version")[1, 1])
}

current_path <- package_location("sceptre")
current_version <- package_version_from_path(current_path)
cat("R executable:", file.path(R.home("bin"), "R"), "\n")
cat("Initial .libPaths():", paste(.libPaths(), collapse = " | "), "\n")
cat("Initially resolved SCEPTRE path:", ifelse(nzchar(current_path), current_path, "<not found>"), "\n")
cat("Initially resolved SCEPTRE version:", as.character(current_version), "\n")

need_install <- current_version != required_version
if (need_install) {
  if (!install_if_needed) {
    stop(
      paste0(
        "Exact SCEPTRE ", required_version_text, " is not available. ",
        "Set INSTALL_PRIVATE_SCEPTRE_IF_NEEDED=True or install ", repo, "@", ref,
        " into ", private_lib, "."
      )
    )
  }
  cat("Installing exact SCEPTRE tag", ref, "into", private_lib, "\n")
  remotes_path <- package_location("remotes")
  if (!nzchar(remotes_path)) {
    install.packages(
      "remotes", repos = cran_repository, lib = private_lib,
      dependencies = c("Depends", "Imports", "LinkingTo")
    )
  }
  remotes::install_github(
    repo = repo,
    ref = ref,
    lib = private_lib,
    dependencies = FALSE,
    upgrade = "never",
    force = TRUE,
    build_vignettes = FALSE,
    quiet = FALSE
  )
}

# Re-resolve after installation. This script never attaches the old namespace.
.libPaths(unique(c(private_lib, .libPaths())))
final_path <- package_location("sceptre")
final_version <- package_version_from_path(final_path)
if (!nzchar(final_path)) stop("SCEPTRE is still unavailable after preflight/install.")
if (final_version != required_version) {
  stop(
    paste0(
      "Expected exact SCEPTRE ", required_version_text,
      " but resolved ", final_version, " at ", final_path, "."
    )
  )
}
private_norm <- normalizePath(private_lib, winslash = "/", mustWork = TRUE)
final_norm <- normalizePath(final_path, winslash = "/", mustWork = TRUE)
if (!startsWith(final_norm, paste0(private_norm, "/"))) {
  warning(
    paste0(
      "Exact SCEPTRE version was found outside the private library at ", final_norm,
      ". The version is acceptable, but the package location differs from the preferred private library."
    )
  )
}
cat("Verified SCEPTRE version:", as.character(final_version), "\n")
cat("Verified SCEPTRE package path:", final_path, "\n")
writeLines(
  c(
    paste0("sceptre_version=", as.character(final_version)),
    paste0("sceptre_path=", final_path),
    paste0("private_library=", private_lib),
    paste0("r_version=", R.version.string)
  ),
  file.path(output_dir, "sceptre_private_library_preflight.txt")
)
