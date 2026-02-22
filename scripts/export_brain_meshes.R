#!/usr/bin/env Rscript
# Export brain mesh surfaces from ggseg3d to parquet for Python

library(arrow)
library(dplyr)

# Check if ggseg3d is installed with mesh data
if (!requireNamespace("ggseg3d", quietly = TRUE)) {
  stop("Please install ggseg3d: remotes::install_github('ggseg/ggseg3d')")
}

library(ggseg3d)

# Function to extract mesh data into a data frame
extract_mesh <- function(mesh, surface_name) {
  rows <- list()

  for (hemi in c("lh", "rh")) {
    hemi_mesh <- mesh[[hemi]]

    rows[[length(rows) + 1]] <- tibble(
      hemi = hemi,
      surface = surface_name,
      vertices_x = list(hemi_mesh$vertices$x),
      vertices_y = list(hemi_mesh$vertices$y),
      vertices_z = list(hemi_mesh$vertices$z),
      faces_i = list(hemi_mesh$faces$i),
      faces_j = list(hemi_mesh$faces$j),
      faces_k = list(hemi_mesh$faces$k)
    )
  }

  bind_rows(rows)
}

# Export each surface type
surfaces <- list(
  pial = ggseg3d:::brain_mesh_pial,
  white = ggseg3d:::brain_mesh_white,
  semi_inflated = ggseg3d:::brain_mesh_semi_inflated
)

# Also get inflated from ggseg.formats if available
if (requireNamespace("ggseg.formats", quietly = TRUE)) {
  lh_inflated <- ggseg.formats::get_brain_mesh("inflated", hemisphere = "lh")
  rh_inflated <- ggseg.formats::get_brain_mesh("inflated", hemisphere = "rh")
  if (!is.null(lh_inflated) && !is.null(rh_inflated)) {
    surfaces$inflated <- list(lh = lh_inflated, rh = rh_inflated)
  }
}

all_meshes <- bind_rows(
  lapply(names(surfaces), function(name) {
    extract_mesh(surfaces[[name]], name)
  })
)

# Write to parquet
output_path <- file.path(
  "src", "ggsegpy", "atlases", "data", "brain_meshes.parquet"
)
write_parquet(all_meshes, output_path)

cat("Exported brain meshes to:", output_path, "\n")
cat("Surfaces:", paste(unique(all_meshes$surface), collapse = ", "), "\n")
cat("Hemispheres:", paste(unique(all_meshes$hemi), collapse = ", "), "\n")
