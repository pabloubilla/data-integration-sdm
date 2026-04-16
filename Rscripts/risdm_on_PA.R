# ============================================================
# PA-only comparison:
#   1) standard binomial GLM
#   2) RISDM PA-only spatial model
# ============================================================

setwd("~/Documents/EcoControl/data-integration-sdm")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(pROC)
  library(reticulate)
  library(terra)
  library(sf)
  library(RISDM)
})

# -----------------------------
# paths / settings
# -----------------------------
data_folder <- "full_data"
species_path <- "data/processed/GeoPlant/full_data"
covariates_path <- "data/processed/GeoPlant/climatic"
output_root <- file.path("output", paste0("integration_geoplant_", data_folder))
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

sp <- 356

# -----------------------------
# data loader
# -----------------------------
load_pa_geoplant_full <- function(species_path, covariates_path, sp) {
  
  X_pa_tr <- reticulate::py_load_object(
    file.path(covariates_path, "pa_train_covariates.pkl")
  )
  
  X_pa_te <- reticulate::py_load_object(
    file.path(covariates_path, "pa_test_covariates.pkl")
  )
  
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
  
  y_from_species_table <- function(df, sp) {
    as.integer(vapply(df$speciesId, function(s) {
      sp %in% parse_species_string(s)
    }, logical(1)))
  }
  
  Y_pa_tr <- y_from_species_table(Y_pa_tr_raw, sp)
  Y_pa_te <- y_from_species_table(Y_pa_te_raw, sp)
  
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

cat("PA split -> train:", nrow(X_pa_tr), ", test:", nrow(X_pa_te), "\n")
cat("Focal species:", sp, "\n")
cat("Train prevalence:", mean(Y_pa_tr), "\n")
cat("Test prevalence:", mean(Y_pa_te), "\n")

if (length(unique(stats::na.omit(Y_pa_tr))) < 2) {
  stop("Training data has fewer than 2 classes for this species.")
}
if (length(unique(stats::na.omit(Y_pa_te))) < 2) {
  stop("Test data has fewer than 2 classes for this species.")
}

# ------------------------------------------------------------
# Keep coordinates for RISDM. Use separate data objects:
#   - glm_* : no coordinates
#   - risdm_* : coordinates retained
# ------------------------------------------------------------
if (!all(c("x", "y") %in% names(X_pa_tr)) || !all(c("x", "y") %in% names(X_pa_te))) {
  stop("RISDM workflow needs x and y columns in both train and test covariate tables.")
}

# covariates used as predictors
drop_nonpredictors <- c("x", "y", "PO")
model_covs <- setdiff(covs, drop_nonpredictors)

# safety: keep only numeric predictors
is_num <- vapply(X_pa_tr[, model_covs, drop = FALSE], is.numeric, logical(1))
model_covs <- model_covs[is_num]

cat("\nPredictor columns used:\n")
print(model_covs)

# -----------------------------
# 1) plain binomial GLM
# -----------------------------
glm_train <- X_pa_tr[, model_covs, drop = FALSE]
glm_test  <- X_pa_te[, model_covs, drop = FALSE]
glm_train$y <- Y_pa_tr

glm_formula <- stats::as.formula(
  paste("y ~", paste(model_covs, collapse = " + "))
)

glm_mod <- glm(glm_formula, data = glm_train, family = binomial())

cat("\n================ GLM summary ================\n")
print(summary(glm_mod))

glm_pred <- predict(glm_mod, newdata = glm_test, type = "response")
glm_auc <- as.numeric(pROC::auc(Y_pa_te, glm_pred))

cat(sprintf("\n[GLM PA-only] AUC on PA_test: %.4f\n", glm_auc))

glm_pred_out <- data.frame(
  observed = Y_pa_te,
  predicted = glm_pred
)

# write_csv(
#   glm_pred_out,
#   file.path(output_root, paste0("predictions_glm_PA_only_species_", sp, ".csv"))
# )

# -----------------------------
# 2) RISDM PA-only spatial model
# -----------------------------
# RISDM needs:
#   - spatial coordinates
#   - a raster covariate stack
#   - a mesh
#   - PA data with response + sampled area
#
# We assume:
#   - x/y are on a regular grid or at least rasterizable with terra::rast(type="xyz")
#   - sampled area is 1 for each PA record
#   - train/test share the same covariate support

# combine train+test to build a complete covariate raster stack
all_xy_covs <- bind_rows(
  X_pa_tr[, c("x", "y", model_covs), drop = FALSE],
  X_pa_te[, c("x", "y", model_covs), drop = FALSE]
) %>%
  distinct(x, y, .keep_all = TRUE) %>%
  arrange(y, x)

# helper: build one raster layer from x/y/z
make_one_raster <- function(df, varname) {
  xyz <- df[, c("x", "y", varname)]
  xyz <- xyz[stats::complete.cases(xyz), , drop = FALSE]
  r <- terra::rast(xyz, type = "xyz")
  names(r) <- varname
  r
}

covar_rasters <- lapply(model_covs, function(v) make_one_raster(all_xy_covs, v))
covars_rast <- rast(covar_rasters)

cat("\n================ Raster stack check ================\n")
print(covars_rast)
print(names(covars_rast))
print(terra::ext(covars_rast))
print(terra::res(covars_rast))

# build PA training object for RISDM
pa_train_df <- X_pa_tr[, c("x", "y", model_covs), drop = FALSE]
pa_train_df$PA <- Y_pa_tr
pa_train_df$transectArea <- 1

# remove rows with missing values in predictors / coords / response
keep_train <- stats::complete.cases(pa_train_df[, c("x", "y", model_covs, "PA", "transectArea")])
pa_train_df <- pa_train_df[keep_train, , drop = FALSE]

cat("\nRISDM training rows after complete-case filter:", nrow(pa_train_df), "\n")
cat("RISDM training prevalence:", mean(pa_train_df$PA), "\n")

# convert to sf points
pa_train_sf <- sf::st_as_sf(pa_train_df, coords = c("x", "y"), crs = NA)

# make mesh from first raster layer
mesh_png <- file.path(output_root, paste0("mesh_species_", sp, ".png"))
png(mesh_png, width = 1000, height = 800)

meshy <- makeMesh(
  ras = covars_rast[[1]],
  max.n = c(1000, 500),
  dep.range = 0.5,
  offset = 10,
  expans.mult = 7.5,
  doPlot = TRUE
)

dev.off()

cat("\n================ Mesh info ================\n")
print(meshy)
cat("Number of mesh vertices:", meshy$n, "\n")

# distribution formula uses the raster/covariate names directly
dist_formula <- stats::as.formula(
  paste("~0 +", paste(model_covs, collapse = " + "))
)

# fit PA-only RISDM
fm_pa <- isdm(
  observationList = list(PAdat = pa_train_sf),
  covars = covars_rast,
  mesh = meshy,
  responseNames = c(PA = "PA"),
  sampleAreaNames = c(PA = "transectArea"),
  distributionFormula = dist_formula,
  artefactFormulas = list(PA = ~1),
  control = list(
    prior.range = c(0.5, 0.1),
    prior.space.sigma = c(2, 0.1),
    coord.names = c("x", "y")
  )
)

cat("\n================ RISDM fitted object ================\n")
print(fm_pa)

cat("\n================ RISDM summary ================\n")
print(summary(fm_pa))

cat("\n================ RISDM fixed effects ================\n")
print(fm_pa$mod$summary.fixed)

cat("\n================ RISDM hyperparameters ================\n")
print(fm_pa$mod$summary.hyperpar)

# -----------------------------
# RISDM prediction on raster
# -----------------------------
# predict.isdm works on raster covariates. We predict over the raster stack,
# then extract test-point values from the predicted probability raster.
risdm_pred_obj <- predict(
  fm_pa,
  covars = covars_rast,
  type = "probability",
  quick = TRUE
)

cat("\n================ RISDM prediction object ================\n")
print(risdm_pred_obj)

# use posterior mean layer if available
# quick=TRUE returns a 5-layer raster with Mean populated
pred_rast <- risdm_pred_obj$field
if (!("Mean" %in% names(pred_rast))) {
  stop("RISDM prediction raster does not contain a 'Mean' layer.")
}

# extract test predictions at test coordinates
test_xy <- X_pa_te[, c("x", "y"), drop = FALSE]
risdm_test_pred <- terra::extract(pred_rast[["Mean"]], test_xy)[, 2]

# keep only test rows with finite predictions
ok_test <- is.finite(risdm_test_pred) & !is.na(Y_pa_te)
risdm_auc <- as.numeric(pROC::auc(Y_pa_te[ok_test], risdm_test_pred[ok_test]))

cat(sprintf("\n[RISDM PA-only] AUC on PA_test (where raster prediction available): %.4f\n", risdm_auc))
cat("RISDM test rows used for AUC:", sum(ok_test), "of", length(Y_pa_te), "\n")

risdm_pred_out <- data.frame(
  observed = Y_pa_te[ok_test],
  predicted = risdm_test_pred[ok_test]
)

write_csv(
  risdm_pred_out,
  file.path(output_root, paste0("predictions_risdm_PA_only_species_", sp, ".csv"))
)

# save probability raster
terra::writeRaster(
  pred_rast,
  filename = file.path(output_root, paste0("risdm_probability_raster_species_", sp, ".tif")),
  overwrite = TRUE
)

# -----------------------------
# save summary
# -----------------------------
summary_all <- data.frame(
  species = sp,
  model = c("GLM_PA_only", "RISDM_PA_only"),
  auc = c(round(glm_auc, 4), round(risdm_auc, 4))
)

write_csv(
  summary_all,
  file.path(output_root, paste0("summary_PA_models_species_", sp, ".csv"))
)

cat("\n================ Final summary ================\n")
print(summary_all)
cat("\nMesh plot saved to:\n", mesh_png, "\n")