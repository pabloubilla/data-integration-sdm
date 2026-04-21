# ============================================================
# PA-only RISDM with BIO1..BIO19 over mainland France
# ============================================================

setwd("~/Documents/EcoControl/data-integration-sdm")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(reticulate)
  library(terra)
  library(RISDM)
  library(rworldmap)
  library(pROC)
  library(sf)
})

# -----------------------------
# paths / settings
# -----------------------------
data_folder      <- "full_data"
species_path     <- "data/processed/GeoPlant/full_data"
covariates_path  <- "data/processed/GeoPlant/climatic"
bioclim_dir      <- "data/raw/GeoPlant/Rasters/BioClimatic_Average_1981-2010"
output_root      <- file.path("output", paste0("integration_geoplant_", data_folder))
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

sp <- 4340 # 88

# -----------------------------
# data loader
# -----------------------------
load_pa_geoplant_full <- function(species_path, covariates_path, sp) {
  
  X_pa_tr <- reticulate::py_load_object(
    file.path(covariates_path, "pa_train_covariates.pkl")
  ) |> as.data.frame()
  
  X_pa_te <- reticulate::py_load_object(
    file.path(covariates_path, "pa_test_covariates.pkl")
  ) |> as.data.frame()
  
  Y_pa_tr <- readr::read_csv(
    file.path(species_path, "pa_train_species.csv"),
    show_col_types = FALSE
  )
  
  Y_pa_te <- readr::read_csv(
    file.path(species_path, "pa_test_species.csv"),
    show_col_types = FALSE
  )
  
  parse_species_string <- function(x) {
    x <- ifelse(is.na(x), "", x)
    x <- trimws(as.character(x))
    if (x == "") integer(0) else as.integer(strsplit(x, "\\s+")[[1]])
  }
  
  add_presence_column <- function(df, sp) {
    df$presence <- as.integer(vapply(
      df$speciesId,
      function(s) sp %in% parse_species_string(s),
      logical(1)
    ))
    df
  }
  
  Y_pa_tr <- add_presence_column(Y_pa_tr, sp)
  Y_pa_te <- add_presence_column(Y_pa_te, sp)
  
  list(
    X_pa_tr = X_pa_tr,
    X_pa_te = X_pa_te,
    Y_pa_tr = Y_pa_tr,
    Y_pa_te = Y_pa_te
  )
}

dat <- load_pa_geoplant_full(
  species_path = species_path,
  covariates_path = covariates_path,
  sp = sp
)

X_pa_tr <- dat$X_pa_tr
X_pa_te <- dat$X_pa_te
Y_pa_tr <- dat$Y_pa_tr
Y_pa_te <- dat$Y_pa_te

Y_pa_tr


# -----------------------------
# 1) France mainland polygon
# -----------------------------
world <- getMap(resolution = "low")
fr    <- world[world$ADMIN == "France", ]
fr_v  <- vect(fr)

fr_mainland_bbox <- ext(-6, 8.5, 42, 51.5)
fr_mainland      <- crop(fr_v, fr_mainland_bbox)



# -----------------------------
# 2) Load BIO1..BIO19 and crop to France
# -----------------------------
bio_files <- file.path(bioclim_dir, paste0("bio", 1:19, ".tif"))

missing_files <- bio_files[!file.exists(bio_files)]
if (length(missing_files) > 0) {
  stop("Missing BIO files:\n", paste(missing_files, collapse = "\n"))
}

bioclim <- rast(bio_files)
names(bioclim) <- paste0("bio", 1:19)

# crs system for BIO layers (should be EPSG:4326, but check)


# align polygon CRS to raster CRS if needed
if (!same.crs(fr_mainland, bioclim)) {
  print("CRS mismatch between France polygon and BIO rasters. Reprojecting polygon...")
  fr_mainland <- project(fr_mainland, crs(bioclim))
}

bioclim_fr <- crop(bioclim, fr_mainland)
bioclim_fr <- mask(bioclim_fr, fr_mainland)


# --------------bio13# -----------------------------
# 3) Keep only PA records inside mainland France
# -----------------------------
pts_pa_tr <- vect(Y_pa_tr, geom = c("lon", "lat"), crs = "EPSG:4326")
pts_pa_te <- vect(Y_pa_te, geom = c("lon", "lat"), crs = "EPSG:4326")

if (!same.crs(pts_pa_tr, fr_mainland)) {
  pts_pa_tr <- project(pts_pa_tr, crs(fr_mainland))
}
if (!same.crs(pts_pa_te, fr_mainland)) {
  pts_pa_te <- project(pts_pa_te, crs(fr_mainland))
}

inside_tr <- relate(pts_pa_tr, fr_mainland, relation = "intersects")
inside_te <- relate(pts_pa_te, fr_mainland, relation = "intersects")

Y_pa_tr <- Y_pa_tr[inside_tr, , drop = FALSE]
Y_pa_te <- Y_pa_te[inside_te, , drop = FALSE]

# optional: quick plot
plot(fr_mainland)
points(
  Y_pa_tr$lon,
  Y_pa_tr$lat,
  col = ifelse(Y_pa_tr$presence == 1, "red", "blue"),
  # pch = .1,
  cex = ifelse(Y_pa_tr$presence == 1, 1, 0.3)
)

plot(fr_mainland)
points(
  Y_pa_te$lon,
  Y_pa_te$lat,
  col = ifelse(Y_pa_tr$presence == 1, "orange", "green"),
  # pch = .1,
  cex = ifelse(Y_pa_tr$presence == 1, 1, 0.3)
)


# -----------------------------
# 4) Build PA training/test tables for RISDM
# -----------------------------
pa_train_df <- Y_pa_tr[, c("lon", "lat", "presence", "areaInM2"), drop = FALSE]
pa_test_df  <- Y_pa_te[, c("lon", "lat", "presence", "areaInM2"), drop = FALSE]

pa_train_df <- pa_train_df[complete.cases(pa_train_df), , drop = FALSE]
pa_test_df  <- pa_test_df[complete.cases(pa_test_df), , drop = FALSE]

# sampled area must be positive
# get mean area and impute missing/zero areas with mean
area_to_impute <- 100*100 #mean(pa_train_df$areaInM2[pa_train_df$areaInM2 > 0], na.rm = TRUE)

# same for train and test
pa_train_df$areaInM2 <- area_to_impute
  
#   ifelse(
#   is.na(pa_train_df$areaInM2) | pa_train_df$areaInM2 <= 0,
#   area_to_impute,
#   pa_train_df$areaInM2
# )
pa_test_df$areaInM2 <- area_to_impute
  
#   ifelse(
#   is.na(pa_test_df$areaInM2) | pa_test_df$areaInM2 <= 0,
#   area_to_impute,
#   pa_test_df$areaInM2
# )



# -----------------------------
# 5) Mesh from cropped BIO stack
# -----------------------------

# # project bioclim_fr to a metric CRS for better mesh construction (e.g., UTM)
target_crs <- "EPSG:2154"  # RGF93 / Lambert-
bioclim_fr_proj <- project(bioclim_fr, target_crs)
fr_mainland_proj <- project(fr_mainland, target_crs)

# change resolution to 10x10
# bioclim_fr_proj <- aggregate(bioclim_fr_proj, fact = 10, fun = mean, na.rm = TRUE)
# disagg(bioclim_fr_proj, fact = 100)
# plot(bioclim_fr_proj)


# 
pts <- st_as_sf(pa_train_df, coords = c("lon", "lat"), crs = "OGC:CRS84")
pts_proj <- st_transform(pts, target_crs)
coords <- st_coordinates(pts_proj)
pa_train_df$x <- coords[,1]
pa_train_df$y <- coords[,2]
# 
# # for test
pts_te <- st_as_sf(pa_test_df, coords = c("lon", "lat"), crs = "OGC:CRS84")
pts_te_proj <- st_transform(pts_te, target_crs)
coords_te <- st_coordinates(pts_te_proj)
pa_test_df$x <- coords_te[,1]
pa_test_df$y <- coords_te[,2]
# 

# meshy_bioclim <- makeMesh(
#   ras = bioclim_fr_proj[[1]],
#   offset = 1,
#   max.edge = c(0.02, 0.05),
#   cutoff = 0.004,
#   expans.mult = 1,
#   dep.range = 0.1,
#   hull.res = 150,          # or larger
#   max.n = c(500, 200),     # set explicitly if you want control
#   doPlot = TRUE
# )
meshy_bioclim_proj <- makeMesh(
  ras = bioclim_fr_proj[[1]],
  max.n = c(3000, 400),
  dep.range = 30000,
  max.edge = c(5000, 20000),      
  # cutoff = 1000,                  
  # offset = 25000,
  # expans.mult = 1,
  # hull.res = 45,
  doPlot = TRUE
)

# -----------------------------
# 6) Formula using all 19 BIO layers
# -----------------------------
bio_covs <- names(bioclim_fr)

dist_formula <- as.formula(
  paste("~ 0 +", paste(bio_covs, collapse = " + "))
)

# -----------------------------
# 7) Fit RISDM
# -----------------------------
# fm_pa <- isdm(
#   observationList   = list(PAdat = pa_train_df),
#   covars            = bioclim_fr_proj,
#   mesh              = meshy_bioclim_proj,
#   responseNames     = c(PA = "presence"),
#   sampleAreaNames   = c(PA = "areaInM2"),
#   distributionFormula = "~ 0 + bio1 + bio2 + bio3",
#   artefactFormulas  = list(PA = ~1),
#   control = list(
#     prior.range       = c(0.5, 0.1),
#     prior.space.sigma = c(2, 0.1),
#     coord.names       = c("x", "y")
#   )
# )

# add a layer to the raster called area with constant value
bioclim_fr_proj$area <- 100*100 # or any large value representing the area of each pixel

pa_train_df$presence <- 0

fm_pa <- isdm(
  observationList   = list(PAdat = pa_train_df),
  covars            = bioclim_fr_proj,
  habitatArea = 'area',
  mesh              = meshy_bioclim_proj,
  responseNames     = c(PA = "presence"),
  sampleAreaNames   = c(PA = "areaInM2"),
  distributionFormula = dist_formula,
  artefactFormulas  = list(PA = ~1),
  control = list(
    prior.range       = c(30000, 0.1),   # example: 50 km, not 0.5 degrees
    prior.space.sigma = c(1, 0.1),       # tune as needed
    coord.names       = c("x", "y")
  )
)

# -----------------------------
# 8) Predict over raster
# -----------------------------

risdm_pred <- predict(
  fm_pa,
  covars = bioclim_fr_proj,
  type   = "link",
  habitatArea = 'area',
  quick  = TRUE
)

pred_mean <- risdm_pred$field[["Mean"]]

plot(pred_mean)

# -----------------------------
# 9) Extract predictions at test points
# -----------------------------
test_pts <- vect(pa_test_df, geom = c("x", "y"), crs = crs(pred_mean))
ext_out  <- terra::extract(pred_mean, test_pts)

pa_test_df$pred <- ext_out[[2]]

# -----------------------------
# 10) Evaluate AUC
# -----------------------------
ok <- is.finite(pa_test_df$pred) & !is.na(pa_test_df$presence)

auc_risdm <- as.numeric(
  pROC::auc(pa_test_df$presence[ok], pa_test_df$pred[ok])
)

cat("Train n:", nrow(pa_train_df), "\n")
cat("Test n:", nrow(pa_test_df), "\n")
cat("Train prevalence:", mean(pa_train_df$presence), "\n")
cat("Test prevalence:", mean(pa_test_df$presence), "\n")
cat(sprintf("RISDM AUC: %.4f\n", auc_risdm))

