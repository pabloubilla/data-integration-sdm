# set to working directory
setwd("~/Documents/EcoControl/data-integration-sdm")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(pROC)
  library(reticulate)
})

# paths
data_folder <- "full_data"
species_path <- "data/processed/GeoPlant/full_data"
covariates_path <- "data/processed/GeoPlant/climatic"
output_root <- file.path("output", paste0("integration_geoplant_", data_folder))
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

# focal species
sp <- 356

load_pa_geoplant_full <- function(species_path, covariates_path, sp) {
  
  # load covariates from pickle
  X_pa_tr <- reticulate::py_load_object(
    file.path(covariates_path, "pa_train_covariates.pkl")
  )
  
  X_pa_te <- reticulate::py_load_object(
    file.path(covariates_path, "pa_test_covariates.pkl")
  )
  
  X_pa_tr <- as.data.frame(X_pa_tr)
  X_pa_te <- as.data.frame(X_pa_te)
  
  # load species tables
  Y_pa_tr_raw <- readr::read_csv(
    file.path(species_path, "pa_train_species.csv"),
    show_col_types = FALSE
  )
  
  Y_pa_te_raw <- readr::read_csv(
    file.path(species_path, "pa_test_species.csv"),
    show_col_types = FALSE
  )
  
  # helper: convert "356 3473 3434" -> c(356, 3473, 3434)
  parse_species_string <- function(x) {
    x <- ifelse(is.na(x), "", x)
    x <- trimws(as.character(x))
    
    if (x == "") {
      integer(0)
    } else {
      as.integer(strsplit(x, "\\s+")[[1]])
    }
  }
  
  # build single-species PA response:
  # 1 if focal species is present in that survey, 0 otherwise
  y_from_species_table <- function(df, sp) {
    as.integer(vapply(df$speciesId, function(s) {
      sp %in% parse_species_string(s)
    }, logical(1)))
  }
  
  Y_pa_tr <- y_from_species_table(Y_pa_tr_raw, sp)
  Y_pa_te <- y_from_species_table(Y_pa_te_raw, sp)
  
  # covariates = all columns except first, matching your Python convention
  covs <- names(X_pa_tr)[-1]
  
  list(
    X_pa_tr = X_pa_tr,
    Y_pa_tr = Y_pa_tr,
    X_pa_te = X_pa_te,
    Y_pa_te = Y_pa_te,
    covs = covs
  )
}

dat <- load_pa_geoplant_full(
  species_path = species_path,
  covariates_path = covariates_path,
  sp = sp
)

X_pa_tr <- dat$X_pa_tr
Y_pa_tr <- dat$Y_pa_tr
X_pa_te <- dat$X_pa_te
Y_pa_te <- dat$Y_pa_te
covs <- dat$covs

cat("PA split → train:", nrow(X_pa_tr), ", test:", nrow(X_pa_te), "\n")
cat("Focal species:", sp, "\n")
cat("Train prevalence:", mean(Y_pa_tr), "\n")
cat("Test prevalence:", mean(Y_pa_te), "\n")

# remove coordinates for now
X_pa_tr <- X_pa_tr %>% select(-any_of(c("x", "y")))
X_pa_te <- X_pa_te %>% select(-any_of(c("x", "y")))
covs <- setdiff(covs, c("x", "y", "PO"))

train_dat <- X_pa_tr[, covs, drop = FALSE]
test_dat  <- X_pa_te[, covs, drop = FALSE]
train_dat$y <- Y_pa_tr

if (length(unique(stats::na.omit(Y_pa_tr))) < 2) {
  stop("Training data has fewer than 2 classes for this species.")
}

if (length(unique(stats::na.omit(Y_pa_te))) < 2) {
  stop("Test data has fewer than 2 classes for this species.")
}

# simple PA model
mod <- glm(y ~ ., data = train_dat, family = binomial())

# predict on PA test set
pred <- predict(mod, newdata = test_dat, type = "response")
auc_pa <- as.numeric(pROC::auc(Y_pa_te, pred))

cat(sprintf("\n[PA-only] AUC on PA_test: %.4f\n", auc_pa))

# save predictions
pred_out <- data.frame(
  observed = Y_pa_te,
  predicted = pred
)

write_csv(
  pred_out,
  file.path(output_root, paste0("predictions_PA_only_species_", sp, ".csv"))
)

# save summary
summary_pa <- data.frame(
  species = sp,
  model = "PA_only",
  auc = round(auc_pa, 4)
)

write_csv(
  summary_pa,
  file.path(output_root, paste0("summary_PA_only_species_", sp, ".csv"))
)

cat("\n=== Summary of Results ===\n")
print(summary_pa)