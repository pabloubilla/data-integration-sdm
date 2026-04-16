suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(pROC)
  library(reticulate)
  library(RISDM)
})

# paths
data_folder <- "full_data"
species_path <- "data/processed/GeoPlant/full_data"
covariates_path <- "data/processed/GeoPlant/climatic"
output_root <- file.path("output", paste0("integration_geoplant_", data_folder))
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

# choose a subset of focal species
species_subset <- c(356, 3473, 3434, 3023, 3192, 1999, 3102, 2728, 289)

load_pa_geoplant_full <- function(species_path, covariates_path) {
  X_pa_tr <- reticulate::py_load_object(
    file.path(covariates_path, "pa_train_covariates.pkl")
  )
  X_pa_te <- reticulate::py_load_object(
    file.path(covariates_path, "pa_test_covariates.pkl")
  )
  
 # X_pa_tr <- log(X_pa_tr+1)
#  X_pa_te <- log(X_pa_te+1)
  
  #sd_X_pa_tr <- sapply(X_pa_tr, sd)
  #mean_X_pa_tr <- sapply(X_pa_tr, mean)
  
  #X_pa_tr <- (X_pa_tr - mean_X_pa_tr)/sd_X_pa_tr
  #X_pa_te <- (X_pa_te - mean_X_pa_tr)/sd_X_pa_tr
  
  X_pa_tr <- as.data.frame(X_pa_tr)
  X_pa_te <- as.data.frame(X_pa_te)
  
  Y_pa_tr_raw <- readr::read_csv(
    file.path(species_path, "pa_train_species.csv"),
    show_col_types = FALSE
  )
  Y_pa_te_raw <- readr::read_csv(
    file.path(species_path, "pa_test_species.csv"),
    show_col_types = FALSE
  )
  
  parse_species_string <- function(x) {
    x <- ifelse(is.na(x), "", x)
    x <- trimws(as.character(x))
    
    if (x == "") {
      integer(0)
    } else {
      as.integer(strsplit(x, "\\s+")[[1]])
    }
  }
  
  species_lists_tr <- lapply(Y_pa_tr_raw$speciesId, parse_species_string)
  species_lists_te <- lapply(Y_pa_te_raw$speciesId, parse_species_string)
  
  covs <- names(X_pa_tr)[-1]
  
  list(
    X_pa_tr = X_pa_tr,
    X_pa_te = X_pa_te,
    species_lists_tr = species_lists_tr,
    species_lists_te = species_lists_te,
    covs = covs
  )
}

make_single_species_response <- function(species_lists, sp) {
  as.integer(vapply(species_lists, function(x) sp %in% x, logical(1)))
}

dat <- load_pa_geoplant_full(
  species_path = species_path,
  covariates_path = covariates_path
)

X_pa_tr <- dat$X_pa_tr
X_pa_te <- dat$X_pa_te
species_lists_tr <- dat$species_lists_tr
species_lists_te <- dat$species_lists_te
covs <- dat$covs

cat("PA split → train:", nrow(X_pa_tr), ", test:", nrow(X_pa_te), "\n")

# remove coordinates for now
X_pa_tr <- X_pa_tr %>% select(-any_of(c("x", "y")))
X_pa_te <- X_pa_te %>% select(-any_of(c("x", "y")))
covs <- setdiff(covs, c("x", "y", "PO"))

results <- vector("list", length(species_subset))

for (i in seq_along(species_subset)) {
  sp <- species_subset[i]
  cat("\nFitting species:", sp, "\n")
  
  Y_pa_tr <- make_single_species_response(species_lists_tr, sp)
  Y_pa_te <- make_single_species_response(species_lists_te, sp)
  
  cat("Train prevalence:", mean(Y_pa_tr), "\n")
  cat("Test prevalence:", mean(Y_pa_te), "\n")
  
  # skip species with only one class in train or test
  if (length(unique(stats::na.omit(Y_pa_tr))) < 2) {
    cat("Skipping: training data has fewer than 2 classes.\n")
    results[[i]] <- data.frame(
      species = sp,
      auc = NA_real_,
      prevalence_train = mean(Y_pa_tr),
      prevalence_test = mean(Y_pa_te)
    )
    next
  }
  
  if (length(unique(stats::na.omit(Y_pa_te))) < 2) {
    cat("Skipping: test data has fewer than 2 classes.\n")
    results[[i]] <- data.frame(
      species = sp,
      auc = NA_real_,
      prevalence_train = mean(Y_pa_tr),
      prevalence_test = mean(Y_pa_te)
    )
    next
  }
  
  train_dat <- X_pa_tr[, covs, drop = FALSE]
  test_dat  <- X_pa_te[, covs, drop = FALSE]
  train_dat$y <- Y_pa_tr
  
  mod <- glm(y ~ ., data = train_dat, family = binomial())
  
  pred <- predict(mod, newdata = test_dat, type = "response")
  auc_pa <- as.numeric(pROC::auc(Y_pa_te, pred))
  
  cat(sprintf("[PA-only] AUC on PA_test: %.4f\n", auc_pa))
  
  results[[i]] <- data.frame(
    species = sp,
    auc = auc_pa,
    prevalence_train = mean(Y_pa_tr),
    prevalence_test = mean(Y_pa_te)
  )
}

results_df <- bind_rows(results)
mean_auc <- mean(results_df$auc, na.rm = TRUE)

cat("\n=== Per-species results ===\n")
print(results_df)

cat(sprintf("\n[PA-only] Mean AUC across subset: %.4f\n", mean_auc))

summary_pa <- data.frame(
  model = "PA_only",
  n_species = length(species_subset),
  n_species_valid = sum(!is.na(results_df$auc)),
  mean_auc = round(mean_auc, 4)
)

write_csv(
  results_df,
  file.path(output_root, "auc_pa_only_species_subset.csv")
)

write_csv(
  summary_pa,
  file.path(output_root, "summary_PA_only_species_subset.csv")
)

cat("\n=== Summary of Results ===\n")
print(summary_pa)