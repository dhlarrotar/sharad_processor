#!/usr/bin/env python3
"""
SHARAD Processing Runner: Mawrth Vallis Example

Interactive example script for running batch SHARAD processing in a well-defined
region. This file contains a configuration dictionary that controls:
- ROI selection and filtering
- radargram data source location
- elevation source (HRSC or MOLA)
- processing steps (radargrams, filtering, maps, profiles, simulations)

Use this as a template: copy and modify the `config` dictionary for other regions.
The first non-empty track-list row must contain `latitude,longitude,roi_size`;
the ROI size is the full width in degrees.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import the main processing function
from sharad_processor import read_roi_from_track_file, run_batch_processing

track_file = project_root / 'src' / 'sharad' / 'mawrth_vallis_tracks.txt'
roi_center, roi_size = read_roi_from_track_file(track_file)

# =============================================================================
# CONFIGURATION - Modify these settings for your region or experiment
# =============================================================================

# Main processing configuration dictionary consumed by run_batch_processing().
# Adjust the following settings depending on task requirements:
# - region identity: a friendly name used in outputs and logs
# - ROI bounds: selected tracks are cropped to this area
# - elevation source: determines map/profiles quality and spatial resolution
# - data path references: location of SHARAD radargrams, track list, and output
# - processing flags: selectively enable/disable stages for faster testing

# Processing configuration for Mawrth Vallis region
config = {
    # Region identification
    'region_name': 'Mawrth Vallis',                    # Display name for this processing run

    # Region of Interest (ROI): center and full width, read from the track-list header.
    # Tracks with footprints outside the ROI are excluded when 'roi_filter'=True.
    'roi_center': roi_center,                          # Read from first row of track_file
    'roi_size': roi_size,                              # Read from first row of track_file

    # Elevation reference used to produce maps and profiles:
    # - 'hrsc': high-resolution local dataset
    # - 'mola': global coarser dataset
    'elevation_source': 'mola',                        # Elevation data provider
    'elevation_file': str(project_root / 'src' / 'mola_dem' / 'mola32.nc'),

    # Data file paths
    'track_file': str(track_file),  # Mawrth Vallis track IDs
    'filepath': str(project_root / 'src' / 'sharad' / 'radargrams'),  # SHARAD radargrams directory
    'output_root': str(project_root / 'output'),  # Output directory root

    # Processing options: set True/False to turn full pipeline stages on or off.
    # Useful to run faster tests (e.g., disable maps/profiles) before full batch.
    'generate_radargrams': False,                       # Load and write raw radargrams from SHARAD files
    'apply_filters': ['log_gabor'], # Filtering sequence; remove or reorder as needed
    'generate_maps': False,                             # Requires valid elevation file; creates ENU/GIS maps
    'generate_profiles': False,                         # Generates elevation profiles along each track
    'generate_simulations': True,                      # Generate clutter outputs, even when radargram plots are disabled
   
    # Additional processing parameters
    'roi_filter': True,                                # True=apply ROI cropping; False=process all listed tracks
    'log_gabor_params': None,                          # Optional override for Log-Gabor parameters
    'mask_vertical_noise': True,                       # Detect and mask persistent bright vertical channels
    'noise_channel_min_excess_db': 8.0,                # Minimum brightness above neighboring columns
    'noise_channel_min_persistence': 0.7,              # Fraction of rows where the excess must persist
    'noise_channel_width': 1                           # Additional channels masked on each side
}

# =============================================================================
# EXAMPLES OF OTHER CONFIGURATIONS
# =============================================================================

# Example: Process a custom track list with minimal processing.
# Its first non-empty row must be `latitude,longitude,roi_size`, followed by track IDs.
# example_track_file = project_root / 'src' / 'sharad' / 'custom_tracks.txt'
# example_roi_center, example_roi_size = read_roi_from_track_file(example_track_file)
# config = {
#     'region_name': 'Custom Tracks',
#     'roi_center': example_roi_center,
#     'roi_size': example_roi_size,
#     'elevation_source': 'hrsc',
#     'elevation_file': str(project_root / 'src' / 'hrsc' / 'dtm_hrsc.tif'),
#     'track_file': str(example_track_file),
#     'filepath': str(project_root / 'src' / 'sharad' / 'radargrams'),
#     'output_root': str(project_root / 'output'),
#     'generate_radargrams': True,
#     'apply_filters': ['gaussian'],  # Only Gaussian filter
#     'generate_maps': False,
#     'generate_profiles': False,
#     'generate_simulations': False,
#     'roi_filter': False,            # Do not crop tracks to the ROI
#     'log_gabor_params': None
# }

# Example: Process a different region (e.g., Utopia Planitia) with MOLA.
# The track file's first non-empty row supplies this region's ROI.
# utopia_track_file = project_root / 'src' / 'sharad' / 'utopia_tracks.txt'
# utopia_roi_center, utopia_roi_size = read_roi_from_track_file(utopia_track_file)
# config = {
#     'region_name': 'Utopia Planitia',
#     'roi_center': utopia_roi_center,
#     'roi_size': utopia_roi_size,
#     'elevation_source': 'mola',     # Use global MOLA data
#     'elevation_file': str(project_root / 'src' / 'mola_dem' / 'mola32.nc'),
#     'track_file': str(utopia_track_file),
#     'filepath': str(project_root / 'src' / 'sharad' / 'radargrams'),
#     'output_root': str(project_root / 'output'),
#     'generate_radargrams': True,
#     'apply_filters': ['gaussian', 'lee', 'log_gabor'],
#     'generate_maps': True,
#     'generate_profiles': True,
#     'generate_simulations': True,
#     'roi_filter': True,
#     'log_gabor_params': {'k': 3, 'nscale': 4, 'mult': 2.5, 'norient': 4, 'softness': 0.8}
# }

# =============================================================================
# RUN PROCESSING
# =============================================================================

if __name__ == "__main__":
    # Entry point: execute batch processing with the current config dictionary.
    # For interactive debugging, set breakpoints before this call or comment out
    # stages in the config dictionary.
    run_batch_processing(config)