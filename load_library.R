# Function: load_libraries
# Description: This function loads a list of required libraries and installs them if they are not already available.

load_libraries <- function() {
  # List of required libraries
  libraries <- c("tidyverse", "raster", "ncdf4", "lubridate", "ggplot2", "RNetCDF", "sf")
  
  # Install and load libraries if they are not already available
  for (lib in libraries) {
    if (!require(lib, character.only = TRUE)) {
      install.packages(lib, dependencies = TRUE)
      library(lib, character.only = TRUE)
    } else {
      library(lib, character.only = TRUE)
    }
  }
}