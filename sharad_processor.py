"""Core SHARAD processing module: loading, filtering, simulation, and batch workflows."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage, fft
from skimage import io, exposure
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import logging
from datetime import datetime
import sys

from mola_data import resolve_mola_file

PROJECT_DIR = Path(__file__).resolve().parent
HRSC_DIR = PROJECT_DIR / 'src' / 'hrsc'
SHARAD_DIR = PROJECT_DIR / 'src' / 'sharad'
MOLA_FILE = PROJECT_DIR / 'src' / 'mola_dem' / 'mola32.nc'


def display_path(path):
    """Return a project-relative path for user-facing messages."""
    project_path = Path(path)
    try:
        return project_path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return project_path.name


def display_message(message):
    """Redact local directory prefixes while preserving diagnostic details."""
    roots = sorted(
        {PROJECT_DIR.resolve(), PROJECT_DIR.parent.resolve(), Path.home().resolve()},
        key=lambda root: len(str(root)),
        reverse=True,
    )
    for root in roots:
        for root_text in (str(root), str(root).replace('\\', '\\\\')):
            message = message.replace(root_text, '.')
    return message


def read_roi_from_track_file(track_file):
    """Read ROI center and width from the first non-empty track-list row."""
    track_file = Path(track_file)
    with track_file.open(encoding='utf-8') as file:
        first_row = next((line.strip() for line in file if line.strip()), '')

    try:
        latitude, longitude, roi_size = map(float, first_row.split(','))
    except ValueError as error:
        raise ValueError(
            f"First non-empty row in {display_path(track_file)} must contain latitude, longitude, and ROI size"
        ) from error

    if roi_size <= 0:
        raise ValueError(
            f"ROI size in {display_path(track_file)} must be greater than zero"
        )

    return [latitude, longitude], roi_size


class SHARADProcessor:
    """
    Modular SHARAD data processor with customizable processing options.

    Usage examples:
        processor = SHARADProcessor()

        # Load HRSC data once
        processor.load_hrsc(HRSC_DIR / 'dtm_hrsc.tif')

        # Process specific tracks with custom options
        processor.process_tracks(['01258301', '01823301'],
                               generate_radargrams=True,
                               apply_filters=['gaussian', 'lee', 'log_gabor'],
                               generate_maps=True,
                               generate_profiles=True)

        # Or process individual components
        processor.generate_radargram('01258301')
        processor.apply_filters_to_track('01258301', ['gaussian', 'lee', 'log_gabor'])
        processor.generate_topographic_map('01258301')
    """

    def __init__(self, create_output_directories=True):
        # Constants
        self.LIGHTSPEED = 299792458  # Speed of light (m/s)
        self.TSPACE = 0.0375e-6  # Time sampling interval (seconds)
        self.crossTrackResolution = 460  # Cross-track resolution (meters)

        # Selected elevation source and its regularly spaced coordinate axes.
        self.elevation_data = None
        self.elevation_latitudes = None
        self.elevation_longitudes = None
        self.elevation_source = None
        self.elevation_loaded = False

        # HRSC aliases retained for existing external workflows.
        self.hrsc_elev = None
        self.hrsc_lon_grid = None
        self.hrsc_lat_grid = None
        self.hrsc_R = None
        self.hrsc_loaded = False

        # Configure paths
        self.config = {
            'filepath': str(SHARAD_DIR / 'radargrams'),
            'datalist_file': str(SHARAD_DIR / 'track_ids.txt'),
            'roi_center': [23.1, 341.2],  # Approximate center of Mawrth Vallis (lat, lon)
            'roi_size': 7,  # ROI size in degrees
            'elevation_source': 'mola',
            'elevation_file': None,
            'mask_vertical_noise': False,
            'noise_channel_min_excess_db': 8.0,
            'noise_channel_min_persistence': 0.7,
            'noise_channel_width': 1,
        }
        self.channel_masks = {}

        self.output_dirs = {
            'root_dir': str(PROJECT_DIR / 'output'),
            'radargrams_dir': str(PROJECT_DIR / 'output' / 'radargrams'),
            'filtered_dir': str(PROJECT_DIR / 'output' / 'filtered'),
            'maps_dir': str(PROJECT_DIR / 'output' / 'maps'),
            'simulations_dir': str(PROJECT_DIR / 'output' / 'simulations'),
            'plots_dir': str(PROJECT_DIR / 'output' / 'power_plots'),
            'logs_dir': str(PROJECT_DIR / 'output' / 'logs')
        }

        if create_output_directories:
            self.create_directories()
    
    def load_mola_data(self, mola_file, roi_center=None, roi_size=None):
        """Load a local MOLA DEM window around the configured ROI."""
        try:
            import netCDF4 as nc
            if roi_center is None:
                roi_center = self.config['roi_center']
            if roi_size is None:
                roi_size = self.config['roi_size']

            center_latitude, center_longitude = roi_center
            margin = max(0.5, roi_size * 0.1)
            latitude = None
            longitude = None

            with nc.Dataset(mola_file) as dataset:
                latitude = np.asarray(dataset.variables['latitude'][:], dtype=float)
                raw_longitude = np.asarray(
                    dataset.variables['longitude'][:],
                    dtype=float,
                )
                altitude_variable = dataset.variables['alt']
                expected_shape = (latitude.size, raw_longitude.size)
                if altitude_variable.shape != expected_shape:
                    raise ValueError(
                        'MOLA altitude dimensions do not match its coordinate axes'
                    )

                longitude_offset = (
                    (raw_longitude - center_longitude + 180) % 360
                ) - 180
                latitude_mask = (
                    np.abs(latitude - center_latitude)
                    <= roi_size / 2 + margin
                )
                longitude_mask = (
                    np.abs(longitude_offset) <= roi_size / 2 + margin
                )
                row_indices = np.flatnonzero(latitude_mask)
                column_indices = np.flatnonzero(longitude_mask)
                if not row_indices.size or not column_indices.size:
                    raise ValueError('The selected ROI is outside the MOLA grid')

                row_indices = row_indices[np.argsort(latitude[row_indices])]
                column_indices = column_indices[
                    np.argsort(longitude_offset[column_indices])
                ]
                row_start, row_stop = row_indices.min(), row_indices.max() + 1
                window = altitude_variable[row_start:row_stop, :]
                window = np.ma.asarray(window)
                window = np.ma.filled(window, np.nan)
                elev = np.take(
                    np.take(window, row_indices - row_start, axis=0),
                    column_indices,
                    axis=1,
                ).astype(float, copy=False)

                lat = latitude[row_indices]
                lon = center_longitude + longitude_offset[column_indices]
                lon_grid, lat_grid = np.meshgrid(lon, lat)

            return elev, lon_grid, lat_grid, None
        except Exception as e:
            logging.error(f"Loading MOLA data: {display_message(str(e))}")
            return None, None, None, None
    
    def load_hrsc_data(self, hrsc_file):
        """Load HRSC (High Resolution Stereo Camera) elevation data with georeferencing information"""
        try:
            import rasterio
            
            with rasterio.open(hrsc_file) as src:
                # Read the first band (elevation data) and convert to float
                hrsc_elev = src.read(1, masked=True).astype(float).filled(np.nan)
                transform = src.transform
                crs = src.crs
                
                # Determine the number of rows and columns in the elevation array
                rows, cols = hrsc_elev.shape
                
                # Generate longitude and latitude coordinate arrays based on the geotransform
                lonHRSC = np.linspace(transform.c, transform.c + transform.a * (cols - 1), cols)
                latHRSC = np.linspace(transform.f + transform.e * (rows - 1), transform.f, rows)
                
                # Sort latitude and longitude arrays in ascending order for proper interpolation
                lat_sort_idx = np.argsort(latHRSC)
                lon_sort_idx = np.argsort(lonHRSC)
                
                latHRSC = latHRSC[lat_sort_idx]
                lonHRSC = lonHRSC[lon_sort_idx]
                hrsc_elev = hrsc_elev[np.ix_(lat_sort_idx, lon_sort_idx)]
                
                # Normalize longitude to 0-360 degree range for consistency with planetary coordinates
                lonHRSC = (lonHRSC + 360) % 360
                
                # Create 2D coordinate grids for longitude and latitude
                hrsc_lon_grid, hrsc_lat_grid = np.meshgrid(lonHRSC, latHRSC)
                
                # Return elevation, grids, and some metadata
                R = {'transform': transform, 'crs': crs}
                
            return hrsc_elev, hrsc_lon_grid, hrsc_lat_grid, R
        except Exception as e:
            logging.error(f"Loading HRSC data: {display_message(str(e))}")
            return None, None, None, None
    
    def create_directories(self):
        """Create all necessary output directories"""
        for dir_path in self.output_dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def load_track_data(self, track_id, lat_min, lat_max, lon_min, lon_max):
        """Load geometry and radargram data for a track with dimension validation"""
        geom_file = Path(self.config['filepath']) / f's_{track_id}_geom.tab'
        
        try:
            # Read and parse geometry file
            with open(geom_file, 'r') as f:
                lines = f.readlines()
            
            # Parse latitude and longitude
            lat, lon = [], []
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    try:
                        lat.append(float(parts[2]))
                        lon.append(float(parts[3]))
                    except (ValueError, IndexError):
                        continue
            
            if not lat:
                logging.error(
                    f"No valid geometry data in {display_path(geom_file)}"
                )
                return None, None, None, None
                
            lat = np.array(lat)
            lon = np.array(lon)
            lon[lon < 0] += 360  # Convert longitude
            
            # Load radargram data first to get its dimensions
            D = self.read_img(track_id)
            if D is None:
                logging.error(f"Could not load radargram for track {track_id}")
                return None, None, None, None
            
            # Ensure geometry and radargram dimensions match
            n_samples = D.shape[1]  # Number of columns in radargram
            if len(lat) != n_samples:
                logging.warning(f"Geometry ({len(lat)}) and radargram ({n_samples}) dimensions don't match. Adjusting geometry.")
                # Truncate or interpolate to match radargram dimensions
                if len(lat) > n_samples:
                    lat = lat[:n_samples]
                    lon = lon[:n_samples]
                else:
                    # Linear interpolation if geometry is shorter
                    x_old = np.linspace(0, 1, len(lat))
                    x_new = np.linspace(0, 1, n_samples)
                    lat = np.interp(x_new, x_old, lat)
                    lon = np.interp(x_new, x_old, lon)
            
            # Compare longitude offsets modulo 360 so ROIs crossing 0/360 work.
            roi_center_lon = (lon_min + lon_max) / 2
            roi_half_width = (lon_max - lon_min) / 2
            lon_offset = (lon - roi_center_lon + 180) % 360 - 180
            in_roi = (
                (lat >= lat_min)
                & (lat <= lat_max)
                & (np.abs(lon_offset) <= roi_half_width)
            )
        
            return lat, lon, in_roi, D
        
        except Exception as e:
            logging.error(
                f"Loading track {track_id}: {display_message(str(e))}"
            )
            return None, None, None, None

    def process_tracks(self):
        """Main processing function for SHARAD tracks: loads data, applies processing, generates outputs"""
        print("Starting SHARAD track processing...")
        if hasattr(self, 'track_ids') and self.track_ids:
            track_ids = self.track_ids
        else:
            # Read track IDs from the configured data file
            track_ids = self.read_track_ids()
        print(f"Found {len(track_ids)} track IDs to process.")
        skipped_tracks = []
        
        # Calculate the geographic boundaries of the region of interest
        lat_min, lat_max, lon_min, lon_max = self.define_roi_bounds()
        print(f"ROI defined: Lat {lat_min:.2f} to {lat_max:.2f}, Lon {lon_min:.2f} to {lon_max:.2f}")
        
        processed_count = 0
        
        for track_id in track_ids:
            print(f"\n--- Processing track ID: {track_id} ---")
            try:
                # Load track geometry coordinates and radargram data, filtering for ROI
                print("Loading track data...")
                lat, lon, in_roi, D = self.load_track_data(track_id, lat_min, lat_max, lon_min, lon_max)
                
                if lat is None:
                    print(f"ERROR: Failed to load data for track {track_id}")
                    skipped_tracks.append(track_id)
                    continue
                
                if not np.any(in_roi):
                    print(f"INFO: No data points in ROI for track ID {track_id}. Skipping.")
                    skipped_tracks.append(track_id)
                    continue
                
                print(f"SUCCESS: Loaded radargram with shape {D.shape}, {np.sum(in_roi)} points in ROI")
                
                # Extract coordinates for points within the ROI only
                lat_roi = lat[in_roi]
                lon_roi = lon[in_roi]
                
                # Clip radargram to ROI boundaries and compute spatial/temporal axes
                print("Preprocessing radargram...")
                D_clipped = self.preprocess_radargram(D, in_roi, track_id)
                x_km = self.compute_cross_track_axis(D_clipped.shape[1])
                y_microseconds = self.compute_vertical_axis(D_clipped.shape[0])
                print(f"SUCCESS: Preprocessed to shape {D_clipped.shape}")
                
                # Find the start and end indices of the continuous ROI segment
                start_index, end_index = self.find_roi_indices(in_roi)
                lat_start, lon_start = lat[start_index], lon[start_index]
                lat_end, lon_end = lat[end_index], lon[end_index]
                
                # Save the processed radargram with geographic metadata
                print("Saving original radargram...")
                output_file = self.save_radargram('Radargram', track_id, D_clipped, x_km, y_microseconds,
                                    (lat_start, lon_start), (lat_end, lon_end))
                print(f"SUCCESS: Saved to {display_path(output_file)}")
                
                # Generate and save clutter simulation data for comparison
                print("Processing simulation data...")
                C = self.process_simulation(track_id, start_index, end_index)
                if C is not None:
                    output_file = self.save_radargram('Cluttergram', track_id, C, x_km, y_microseconds,
                                        (lat_start, lon_start), (lat_end, lon_end),
                                        directory='simulations')
                    print(
                        f"SUCCESS: Saved simulation to {display_path(output_file)}"
                    )
                else:
                    print("WARNING: Simulation processing failed")
                
                # Apply speckle reduction filters and save filtered versions
                print("Applying filters...")
                self.apply_filters_and_save(track_id, D_clipped, x_km, y_microseconds,
                                          (lat_start, lon_start), (lat_end, lon_end))
                print("SUCCESS: Filters applied and saved")
                
                # Generate power vs time plots for analysis
                print("Creating power vs time plots...")
                self.plot_power_vs_time(track_id)
                print("SUCCESS: Power plots created")
                
                # Create topographic map overlay for this track
                print("Creating topographic map...")
                self.create_topographic_map(track_id)
                print("SUCCESS: Topographic map created")
                
                # Generate elevation profile along the track
                print("Creating elevation profile...")
                self.create_elevation_profile(track_id)
                print("SUCCESS: Elevation profile created")
                
                processed_count += 1
                print(f"COMPLETED: Track {track_id} processed successfully")
                
            except Exception as e:
                print(
                    f"ERROR: Processing track ID {track_id} failed: "
                    f"{display_message(str(e))}"
                )
                skipped_tracks.append(track_id)
        
        # Save skipped tracks list
        self.save_skipped_tracks(skipped_tracks)
        print(f"\n--- Processing Summary ---")
        print(f"Total tracks: {len(track_ids)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Skipped/failed: {len(skipped_tracks)}")
        if skipped_tracks:
            print(f"Skipped tracks: {', '.join(skipped_tracks)}")
        print(
            "Outputs saved to: "
            f"{display_path(self.output_dirs['root_dir'])}"
        )
        print("Processing complete.")
        print(f"\n--- Processing Summary ---")
        print(f"Total tracks: {len(track_ids)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Skipped/failed: {len(skipped_tracks)}")
        if skipped_tracks:
            print(f"Skipped tracks: {', '.join(skipped_tracks)}")
        print(
            "Outputs saved to: "
            f"{display_path(self.output_dirs['root_dir'])}"
        )
        print("Processing complete.")
    
    def read_track_ids(self):
        """Read the list of track IDs from a file"""
        with open(self.config['datalist_file'], 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Check if first line contains ROI parameters (lat,lon,size format)
        if lines and ',' in lines[0]:
            try:
                roi_params = lines[0].split(',')
                if len(roi_params) == 3:
                    # Try to parse as floats - if successful, it's ROI parameters
                    [float(x) for x in roi_params]
                    # Skip the first line (ROI parameters) and return the rest
                    return lines[1:]
            except ValueError:
                pass  # Not ROI parameters, treat as track IDs
        
        return lines
    
    def define_roi_bounds(self):
        """Define ROI bounds based on center and size"""
        lat_min = self.config['roi_center'][0] - self.config['roi_size'] / 2
        lat_max = self.config['roi_center'][0] + self.config['roi_size'] / 2
        lon_min = self.config['roi_center'][1] - self.config['roi_size'] / 2
        lon_max = self.config['roi_center'][1] + self.config['roi_size'] / 2
        return lat_min, lat_max, lon_min, lon_max
    
    
    
    def read_img(self, track_id):
        """Read SHARAD radargram image using metadata from .lbl file"""
        img_file = Path(self.config['filepath']) / f's_{track_id}_rgram.img'
        lbl_file = Path(self.config['filepath']) / f's_{track_id}_rgram.lbl'
        
        # Parse .lbl file for dimensions
        lines = None
        samples = None
        with open(lbl_file, 'r') as f:
            for line in f:
                if 'LINES' in line and '=' in line:
                    lines = int(line.split('=')[1].strip())
                elif 'LINE_SAMPLES' in line and '=' in line:
                    samples = int(line.split('=')[1].strip())
        
        if lines is None or samples is None:
            logging.error(
                f"Could not parse dimensions from {display_path(lbl_file)}"
            )
            return None
        
        try:
            with open(img_file, 'rb') as f:
                data = np.fromfile(f, dtype='<f4')
            
            # Reshape based on metadata
            D = data.reshape((lines, samples))
            logging.debug(
                f"Loaded radargram {img_file.name} with shape ({lines}, {samples})"
            )
            return D
        
        except Exception as e:
            logging.error(
                f"Reading image file {display_path(img_file)}: "
                f"{display_message(str(e))}"
            )
            return None
    
    def preprocess_radargram(self, D, in_roi, track_id=None):
        """Preprocess radargram data with contrast stretching"""
        D_clipped = D[:, in_roi]
        
        # Convert to dB scale
        D_clipped = 10 * np.log10(
            np.maximum(D_clipped, np.finfo(float).tiny)
        )
        if track_id is not None:
            if self.config.get('mask_vertical_noise', False):
                channel_mask = self.detect_vertical_noise_channels(D_clipped)
                self.channel_masks[track_id] = channel_mask
                masked_count = int(np.count_nonzero(channel_mask))
                if masked_count:
                    logging.info(
                        f'Masked {masked_count} vertical noise channel(s) '
                        f'for track {track_id}'
                    )
                    print(
                        f'   Masked {masked_count} vertical noise channel(s) '
                        f'for track {track_id}'
                    )
            else:
                self.channel_masks.pop(track_id, None)
        return D_clipped

    def detect_vertical_noise_channels(self, radargram_db):
        """Detect persistently bright vertical columns against local neighbors."""
        radargram_db = np.asarray(radargram_db, dtype=float)
        if radargram_db.ndim != 2 or radargram_db.shape[1] < 5:
            return np.zeros(
                radargram_db.shape[1] if radargram_db.ndim == 2 else 0,
                dtype=bool,
            )

        minimum_excess_db = float(
            self.config.get('noise_channel_min_excess_db', 8.0)
        )
        minimum_persistence = float(
            self.config.get('noise_channel_min_persistence', 0.7)
        )
        expansion_width = int(self.config.get('noise_channel_width', 1))
        if minimum_excess_db <= 0:
            raise ValueError('Minimum noise-channel excess must be positive')
        if not 0 < minimum_persistence <= 1:
            raise ValueError('Noise-channel persistence must be between 0 and 1')
        if expansion_width < 0:
            raise ValueError('Noise-channel expansion width cannot be negative')

        row_stride = max(1, int(np.ceil(radargram_db.shape[0] / 512)))
        sampled = radargram_db[::row_stride, :]
        finite_values = sampled[np.isfinite(sampled)]
        if not finite_values.size:
            return np.zeros(sampled.shape[1], dtype=bool)
        sampled = np.where(np.isfinite(sampled), sampled, np.median(finite_values))
        local_baseline = ndimage.median_filter(
            sampled,
            size=(1, 7),
            mode='nearest',
        )
        excess = sampled - local_baseline
        positive_excess = np.maximum(excess, 0)
        column_excess = np.median(positive_excess, axis=0)
        persistence = np.mean(excess >= minimum_excess_db, axis=0)

        noisy_channels = (
            (column_excess >= minimum_excess_db)
            & (persistence >= minimum_persistence)
        )
        if expansion_width:
            noisy_channels = ndimage.maximum_filter1d(
                noisy_channels.astype(np.uint8),
                size=2 * expansion_width + 1,
                mode='constant',
            ).astype(bool)
        return noisy_channels

    def apply_channel_mask(self, track_id, data):
        """Return radargram data with detected channels masked for plotting."""
        channel_mask = self.channel_masks.get(track_id)
        if channel_mask is None or len(channel_mask) != data.shape[1]:
            return data
        return np.ma.array(
            data,
            mask=np.broadcast_to(channel_mask, data.shape),
            copy=False,
        )
    
    def compute_cross_track_axis(self, num_columns):
        """Compute cross-track axis in kilometers"""
        return np.arange(num_columns) * self.crossTrackResolution / 1000
    
    def compute_vertical_axis(self, num_rows):
        """Compute vertical axis in microseconds"""
        return np.arange(num_rows) * self.TSPACE * 1e6
    
    def find_roi_indices(self, in_roi):
        """Find the start and end indices of the ROI"""
        roi_indices = np.where(in_roi)[0]
        start_index = roi_indices[0]
        end_index = roi_indices[-1]
        return start_index, end_index
    
    def save_radargram(self, title, track_id, data, x_km, y_microseconds, 
                      start_coords, end_coords, directory='radargrams'):
        """Save radargram with dynamically adjusted y-axis"""
        # Compute dynamic y_min_us based on maximum power locations
        max_times = []
        channel_mask = (
            self.channel_masks.get(track_id)
            if title.lower() == 'radargram' and directory == 'radargrams'
            else None
        )
        valid_columns = (
            np.flatnonzero(~channel_mask)
            if channel_mask is not None and len(channel_mask) == data.shape[1]
            else range(data.shape[1])
        )
        for col in valid_columns:
            if data.shape[0] > 0:
                col_max_idx = np.argmax(data[:, col])
                max_times.append(y_microseconds[col_max_idx])
        
        if max_times:
            # Use 5th percentile of max times to avoid outliers, with buffer
            y_min_us = max(0, np.percentile(max_times, 5) - 10)
        else:
            y_min_us = 60  # fallback
        
        # Set y_max_us dynamically as well
        if max_times:
            y_max_us = min(150, np.percentile(max_times, 95) + 20)  # Cap at 150 us
        else:
            y_max_us = 130  # fallback
        
        # Apply y-axis cropping
        valid_rows = (y_microseconds >= y_min_us) & (y_microseconds <= y_max_us)
        data = data[valid_rows, :]
        y_microseconds = y_microseconds[valid_rows]
        if title.lower() == 'radargram' and directory == 'radargrams':
            data = self.apply_channel_mask(track_id, data)
        
        output_dir = self.output_dirs[f'{directory}_dir']
        output_file = Path(output_dir) / f'{title}_{track_id}.png'
        
        # Determine unit for plotting
        unit = 'simulation' if title.lower() in ['cluttergram', 'simulation'] else 'dB'
        
        self.plot_data(data, x_km, y_microseconds, str(output_file), 
                      f'{title} (Track ID: {track_id})',
                      start_coords, end_coords, unit)
        
        return str(output_file)
    
    def apply_filters_and_save(self, track_id, D_clipped, x_km, y_microseconds,
                             start_coords, end_coords):
        """Apply filters and save filtered radargrams"""
        sigma = 1  # Gaussian filter sigma value
        
        # Apply Gaussian filter
        print("  Applying Gaussian filter...")
        D_gaussian = ndimage.gaussian_filter(D_clipped, sigma=sigma)
        
        # Apply Lee filter
        print("  Applying Lee filter...")
        D_Lee = self.lee_filter(D_clipped, n=3, ENL=2.5, function_mode='r')
        
        # Apply Log-Gabor filter
        print("  Applying Log-Gabor filter...")
        D_logGabor = self.noisecomp(D_clipped, k=5, nscale=6, mult=3, 
                                   norient=6, softness=0.6)
        
        # Save filtered radargrams
        print("  Saving filtered radargrams...")
        self.save_filtered_radargram(track_id, D_gaussian, D_Lee, D_logGabor,
                                    x_km, y_microseconds, sigma,
                                    start_coords, end_coords)
    
    def save_filtered_radargram(self, track_id, D_gaussian, D_Lee, D_logGabor,
                              x_km, y_microseconds, sigma,
                              start_coords, end_coords):
        """Save filtered radargrams with dynamically adjusted y-axis"""
        # Compute dynamic y-axis range based on the first non-None filtered data
        reference_data = D_gaussian if D_gaussian is not None else (D_Lee if D_Lee is not None else D_logGabor)
        
        if reference_data is not None:
            max_times = []
            channel_mask = self.channel_masks.get(track_id)
            valid_columns = (
                np.flatnonzero(~channel_mask)
                if channel_mask is not None
                and len(channel_mask) == reference_data.shape[1]
                else range(reference_data.shape[1])
            )
            for col in valid_columns:
                if reference_data.shape[0] > 0:
                    col_max_idx = np.argmax(reference_data[:, col])
                    max_times.append(y_microseconds[col_max_idx])
            
            if max_times:
                y_min_us = max(0, np.percentile(max_times, 5) - 10)
                y_max_us = min(150, np.percentile(max_times, 95) + 20)
            else:
                y_min_us = 60
                y_max_us = 130
        else:
            y_min_us = 60
            y_max_us = 130
        
        # Apply y-axis cropping
        valid_rows = (y_microseconds >= y_min_us) & (y_microseconds <= y_max_us)
        y_microseconds = y_microseconds[valid_rows]
        
        # Apply y-axis cropping only to non-None arrays
        if D_gaussian is not None:
            D_gaussian = D_gaussian[valid_rows, :]
        if D_Lee is not None:
            D_Lee = D_Lee[valid_rows, :]
        if D_logGabor is not None:
            D_logGabor = D_logGabor[valid_rows, :]

        if D_gaussian is not None:
            D_gaussian = self.apply_channel_mask(track_id, D_gaussian)
        if D_Lee is not None:
            D_Lee = self.apply_channel_mask(track_id, D_Lee)
        if D_logGabor is not None:
            D_logGabor = self.apply_channel_mask(track_id, D_logGabor)
        
        saved_files = []

        # Save each requested filtered radargram
        if D_gaussian is not None:
            output_file = Path(self.output_dirs['filtered_dir']) / f'filtered_gaussian_sigma{sigma}_{track_id}.png'
            self.plot_data(D_gaussian, x_km, y_microseconds, str(output_file),
                          f'Gaussian Filtered Radargram (Track ID: {track_id})',
                          start_coords, end_coords, 'dB')
            print(f"    Saved Gaussian: {display_path(output_file)}")
            saved_files.append(str(output_file))
        
        if D_logGabor is not None:
            output_file = Path(self.output_dirs['filtered_dir']) / f'filtered_logGabor_sigma{sigma}_{track_id}.png'
            self.plot_data(D_logGabor, x_km, y_microseconds, str(output_file),
                          f'Log-Gabor Filtered Radargram (Track ID: {track_id})',
                          start_coords, end_coords, 'dB')
            print(f"    Saved Log-Gabor: {display_path(output_file)}")
            saved_files.append(str(output_file))
        
        if D_Lee is not None:
            output_file = Path(self.output_dirs['filtered_dir']) / f'filtered_Lee_d_{track_id}.png'
            self.plot_data(D_Lee, x_km, y_microseconds, str(output_file),
                          f'Lee Filtered Radargram (Track ID: {track_id})',
                          start_coords, end_coords, 'dB')
            print(f"    Saved Lee: {display_path(output_file)}")
            saved_files.append(str(output_file))

        return saved_files[0] if saved_files else None
    
    def plot_data(self, data, x_km, y_microseconds, filename, title_str,
                 start_coords, end_coords, unit):
        """Plot radargram data"""
        # Larger figure for poster
        plt.figure(figsize=(20, 10))
        
        # Larger fonts for readability
        FONT_SIZE_TITLE = 32
        FONT_SIZE_LABEL = 28
        
        # Plot the data
        if unit == 'simulation':
            cmap = plt.get_cmap('viridis').copy()
            vmin = -35
            vmax = -27
        else:
            cmap = plt.get_cmap('turbo').copy()
            vmin = None
            vmax = None
        cmap.set_bad('black')
        
        plt.gca().set_facecolor('black')
        plt.imshow(data, aspect='auto', extent=[x_km[0], x_km[-1], 
                                               y_microseconds[-1], y_microseconds[0]],
                  cmap=cmap, vmin=vmin, vmax=vmax)
        
        if unit == 'dB':
            cbar = plt.colorbar()
            cbar.set_label('Radar Backscatter Power (dB)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
            cbar.ax.tick_params(labelsize=FONT_SIZE_LABEL)
        elif unit == 'simulation':
            cbar = plt.colorbar()
            cbar.set_label('Simulation Backscatter (ln)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
            cbar.ax.tick_params(labelsize=FONT_SIZE_LABEL)
        else:
            # For Log-Gabor and other filtered data
            cbar = plt.colorbar()
            cbar.set_label('Filtered Backscatter', fontsize=FONT_SIZE_LABEL, fontweight='bold')
            cbar.ax.tick_params(labelsize=FONT_SIZE_LABEL)
        
        plt.xlabel('Cross-Track Distance (km)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        plt.ylabel('Time Delay (µs)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        plt.title(title_str, fontsize=FONT_SIZE_TITLE, fontweight='bold')
        plt.xticks(fontsize=FONT_SIZE_LABEL)
        plt.yticks(fontsize=FONT_SIZE_LABEL)
        
        # Add start and end coordinates
        if start_coords:
            plt.text(x_km[0], y_microseconds[0] + 7 - 0.035 * (y_microseconds[-1] - y_microseconds[0]),
                    f' A ({start_coords[0]:.2f}°N, {start_coords[1]:.2f}°E)',
                    color='white', fontsize=FONT_SIZE_LABEL, fontweight='bold', ha='left')
        
        if end_coords:
            plt.text(x_km[-1], y_microseconds[0] + 7 - 0.035 * (y_microseconds[-1] - y_microseconds[0]),
                    f'({end_coords[0]:.2f}°N, {end_coords[1]:.2f}°E) A´',
                    color='white', fontsize=FONT_SIZE_LABEL, fontweight='bold', ha='right')
        
        # Add scale bars
        self.add_scale_bars(x_km, y_microseconds, FONT_SIZE_LABEL)
        
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    def add_scale_bars(self, x_km, y_microseconds, FONT_SIZE_LABEL):
        """Add horizontal and vertical scale bars to the plot"""
        x_scale_length = 20  # Horizontal scale length in km
        x_scale_start = x_km[0] + 0.1 * (x_km[-1] - x_km[0])
        y_scale_pos = y_microseconds[-1] - 0.1 * (y_microseconds[-1] - y_microseconds[0])
        
        plt.plot([x_scale_start, x_scale_start + x_scale_length], 
                 [y_scale_pos, y_scale_pos], 'w-', linewidth=3.5)
        plt.text(x_scale_start + x_scale_length / 2, 
                 y_scale_pos - 0.035 * (y_microseconds[-1] - y_microseconds[0]),
                 f'{x_scale_length} km', color='white', fontsize=FONT_SIZE_LABEL, ha='center')
        
        y_scale_length_us = 10  # Vertical scale length in µs
        x_scale_pos = x_km[-1] - 0.1 * (x_km[-1] - x_km[0])
        y_scale_start = y_microseconds[0] + 0.7 * (y_microseconds[-1] - y_microseconds[0])
        
        plt.plot([x_scale_pos, x_scale_pos], 
                 [y_scale_start, y_scale_start + y_scale_length_us], 
                 'w-', linewidth=3.5)
        
        # Scale bar labels
        plt.text(x_scale_pos - 0.035 * (x_km[-1] - x_km[0]), 
                 y_scale_start + y_scale_length_us / 2,
                 f'{y_scale_length_us} µs', color='white', fontsize=FONT_SIZE_LABEL, 
                 fontweight='bold', ha='right', rotation=90)
        
        y_scale_length_km = y_scale_length_us * self.LIGHTSPEED / (2 * 1e9)
        plt.text(x_scale_pos + 0.035 * (x_km[-1] - x_km[0]), 
                 y_scale_start + y_scale_length_us / 2,
                 f'≈ {y_scale_length_km:.2f} km', color='white', fontsize=FONT_SIZE_LABEL, 
                 fontweight='bold', ha='left', rotation=90)
    
    def process_simulation(self, track_id, start_col, end_col):
        """Process simulation data for a given track ID - exact MATLAB implementation"""
        cluttergram_dir = Path(self.config['filepath']).parent / 'cluttergrams'
        sim_file = cluttergram_dir / f's_{track_id}_sim.img'
        xml_file = cluttergram_dir / f's_{track_id}_sim.xml'
        
        # Parse XML exactly like MATLAB
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        sampleCounts = []
        lineCounts = []
        
        for array in root.findall('.//{http://pds.nasa.gov/pds4/pds/v1}Array_2D_Image'):
            lineValue = None
            sampleValue = None
            for axis in array.findall('.//{http://pds.nasa.gov/pds4/pds/v1}Axis_Array'):
                axis_name = axis.find('.//{http://pds.nasa.gov/pds4/pds/v1}axis_name')
                elements = axis.find('.//{http://pds.nasa.gov/pds4/pds/v1}elements')
                if axis_name is not None and elements is not None:
                    if axis_name.text == 'Line':
                        lineValue = int(elements.text)
                    elif axis_name.text == 'Sample':
                        sampleValue = int(elements.text)
            if lineValue is not None and sampleValue is not None:
                lineCounts.append(lineValue)
                sampleCounts.append(sampleValue)
        
        if not lineCounts or not sampleCounts:
            print(f"Could not parse XML for {track_id}")
            return None
        
        # Use the last (Combined) dimensions
        num_lines = lineCounts[-1]
        num_samples = sampleCounts[-1]
        
        # Calculate offsets like MATLAB
        offsets = [0, num_lines * num_samples * 4, 2 * num_lines * num_samples * 4]
        
        C = None
        datasets = [
            {'name': 'Left Side', 'offset': offsets[0]},
            {'name': 'Right Side', 'offset': offsets[1]},
            {'name': 'Combined', 'offset': offsets[2]}
        ]
        
        for dataset in datasets:
            try:
                with open(sim_file, 'rb') as f:
                    f.seek(dataset['offset'])
                    # Read [num_samples, num_lines] like MATLAB
                    count = num_samples * num_lines
                    data_flat = np.fromfile(f, dtype='<f4', count=count)
                    if len(data_flat) != count:
                        continue
                    # Reshape to [num_samples, num_lines] - MATLAB fills column-major
                    data = data_flat.reshape((num_samples, num_lines), order='F')  # order='F' for column-major
                    
                    # Extract ROI
                    row_start = 0  # 0-based
                    row_end = 3600
                    col_start = start_col  # 0-based
                    col_end = end_col  # 0-based
                    
                    data = data[col_start:col_end+1, row_start:row_end]
                    
                    # Apply log(data')
                    data_T = data.T
                    data_logged = np.log(np.maximum(data_T, 1e-20))
                    
                    if dataset['name'] == 'Combined':
                        C = data_logged
                        
            except Exception as e:
                print(
                    f"Error reading {dataset['name']}: "
                    f"{display_message(str(e))}"
                )
                continue
        
        if C is None:
            logging.warning(f"Could not read Combined simulation data for {track_id}")
            return None
        
        logging.debug(f"Simulation C shape {C.shape}, min {np.min(C):.3f}, max {np.max(C):.3f}")
        return C
    
    def save_skipped_tracks(self, skipped_tracks):
        """Save the list of skipped tracks to a file"""
        skipped_file = Path(self.output_dirs['logs_dir']) / 'skipped_tracks.txt'
        with open(skipped_file, 'w') as f:
            for track_id in skipped_tracks:
                f.write(f'{track_id}\n')
    
    def set_hrsc_data(self, hrsc_elev, hrsc_lon_grid, hrsc_lat_grid, R):
        """Set HRSC data for use in visualizations"""
        self.set_elevation_data(hrsc_elev, hrsc_lon_grid, hrsc_lat_grid, R, 'hrsc')

    def set_elevation_data(self, elevation, longitude_grid, latitude_grid, metadata, source):
        """Store elevation data with ascending axes and longitudes near the ROI."""
        latitudes = np.asarray(latitude_grid[:, 0], dtype=float)
        center_longitude = float(self.config['roi_center'][1])
        longitudes = center_longitude + (
            (np.asarray(longitude_grid[0, :], dtype=float) - center_longitude + 180)
            % 360
        ) - 180
        latitude_order = np.argsort(latitudes)
        longitude_order = np.argsort(longitudes)
        self.elevation_data = np.asarray(elevation, dtype=float)[
            np.ix_(latitude_order, longitude_order)
        ]
        self.elevation_latitudes = latitudes[latitude_order]
        self.elevation_longitudes = longitudes[longitude_order]
        self.elevation_source = source.lower()
        self.elevation_loaded = True
        self.hrsc_loaded = self.elevation_source == 'hrsc'

        if self.hrsc_loaded:
            self.hrsc_elev = self.elevation_data
            self.hrsc_lon_grid, self.hrsc_lat_grid = np.meshgrid(
                self.elevation_longitudes,
                self.elevation_latitudes,
            )
            self.hrsc_R = metadata

        print(f"{self.elevation_source.upper()} data stored for visualization use")

    def clear_elevation_data(self):
        """Clear loaded elevation state before attempting another source."""
        self.elevation_data = None
        self.elevation_latitudes = None
        self.elevation_longitudes = None
        self.elevation_source = None
        self.elevation_loaded = False
        self.hrsc_elev = None
        self.hrsc_lon_grid = None
        self.hrsc_lat_grid = None
        self.hrsc_R = None
        self.hrsc_loaded = False
    
    # Filter implementations
    def lee_filter(self, Im, n=3, ENL=2.5, function_mode='r'):
        """
        Python implementation of Lee filter
        Parameters:
            Im: Input image
            n: Number of pixels on each side of the center pixel (use 3 or 4)
            ENL: Equivalent number of looks (mean value from homogeneous regions)
            function_mode: 'r' for ratio or 'd' for difference filtering
        """
        sigma_v = np.sqrt(1/ENL)
        
        # Calculate local mean and variance
        E_norm = ndimage.uniform_filter(Im, size=2*n+1)
        Var_y_norm = ndimage.uniform_filter(Im**2, size=2*n+1) - E_norm**2
        
        # Calculate weight
        Var_x_norm = (Var_y_norm - E_norm**2 * sigma_v**2) / (1 + sigma_v**2)
        Var_x_norm[Var_x_norm < 0] = 0
        k_norm = Var_x_norm / (Var_y_norm + 1e-10)  # Add small value to avoid division by zero
        k_norm[k_norm < 0] = 0
        
        # Apply filter
        Im_filtered = E_norm + k_norm * (Im - E_norm)
        
        # Handle edges (simplified)
        Im_filtered[:n, :] = Im[:n, :]
        Im_filtered[-n:, :] = Im[-n:, :]
        Im_filtered[:, :n] = Im[:, :n]
        Im_filtered[:, -n:] = Im[:, -n:]
        
        return Im_filtered
    
    def noisecomp(self, image, k=5, nscale=6, mult=3, norient=6, softness=0.6):
        """
        Python implementation of Log-Gabor filter based noise reduction
        Parameters:
            image: Input image
            k: No of standard deviations of noise to reject (2-3)
            nscale: No of filter scales to use (5-7)
            mult: Multiplying factor between scales (2.5-3)
            norient: No of orientations to use (6)
            softness: Degree of soft thresholding (0-hard, 1-soft)
        """
        minWaveLength = 2
        sigmaOnf = 0.55
        dThetaOnSigma = 1.0
        epsilon = 1e-5
        
        thetaSigma = np.pi/norient/dThetaOnSigma
        imagefft = fft.fft2(image)
        rows, cols = imagefft.shape
        
        # Create frequency domain grids
        x = np.ones((rows, 1)) * np.linspace(-cols/2, cols/2-1, cols)/(cols/2)
        y = np.linspace(-rows/2, rows/2-1, rows).reshape(-1, 1) * np.ones((1, cols))/(rows/2)
        
        radius = np.sqrt(x**2 + y**2)
        radius[int(rows/2), int(cols/2)] = 1  # Avoid division by zero
        theta = np.arctan2(-y, x)
        
        totalEnergy = np.zeros((rows, cols))
        
        for o in range(norient):
            angl = o * np.pi/norient
            
            # Calculate angular distance
            ds = np.sin(theta) * np.cos(angl) - np.cos(theta) * np.sin(angl)
            dc = np.cos(theta) * np.cos(angl) + np.sin(theta) * np.sin(angl)
            dtheta = np.abs(np.arctan2(ds, dc))
            spread = np.exp((-dtheta**2) / (2 * thetaSigma**2))
            
            wavelength = minWaveLength
            for s in range(nscale):
                fo = 1.0/wavelength
                rfo = fo/0.5
                logGabor = np.exp((-(np.log(radius/rfo))**2) / (2 * np.log(sigmaOnf)**2))
                logGabor[int(rows/2), int(cols/2)] = 0
                
                filter = logGabor * spread
                filter = fft.fftshift(filter)
                
                # Convolve image with filter
                EOfft = imagefft * filter
                EO = np.real(fft.ifft2(EOfft))
                aEO = np.abs(EO)
                
                if s == 0:
                    # Estimate noise statistics
                    medianEn = np.median(aEO.flatten())
                    meanEn = medianEn * 0.5 * np.sqrt(-np.pi/np.log(0.5))
                    RayVar = (4-np.pi) * (meanEn**2)/np.pi
                    RayMean = meanEn
                
                # Apply soft thresholding
                T = (RayMean + k*np.sqrt(RayVar))/(mult**s)
                validEO = aEO > T
                V = softness * T * EO / (aEO + epsilon)
                V = ~validEO * EO + validEO * V
                EO = EO - V
                
                totalEnergy += EO
                wavelength *= mult
        
        return np.real(totalEnergy)

    def create_joint_visualizations(self):
        """Create joint visualizations for all processed tracks"""
        print("Creating joint visualizations...")
        
        # Get track IDs from processed radargrams
        radargram_dir = Path('./output/radargrams')
        if not radargram_dir.exists():
            print("No radargrams found. Skipping joint visualizations.")
            return
            
        track_files = list(radargram_dir.glob('*_radargram.png'))
        if not track_files:
            track_files = list(radargram_dir.glob('Radargram_*.png'))
            
        print(f"Found {len(track_files)} radargram files: {[f.name for f in track_files]}")
        
        track_ids = []
        for f in track_files:
            # Extract track ID from filename like "Radargram_01258301.png"
            parts = f.stem.split('_')
            if len(parts) >= 2:
                track_id = parts[-1]
                track_ids.append(track_id)
                print(f"Extracted track ID: {track_id} from {f.name}")
        
        print(f"Found {len(track_ids)} tracks to create joint visualizations for: {track_ids}")
        
        for track_id in track_ids:
            try:
                self.create_joint_visualization(track_id)
                print(f"Created joint visualization for track {track_id}")
            except Exception as e:
                print(
                    f"Error creating joint visualization for track {track_id}: "
                    f"{display_message(str(e))}"
                )
                import traceback
                print(display_message(traceback.format_exc()))

    def create_joint_visualization(self, track_id):
        """Create joint visualization with radargram, cluttergram, topographic map, and elevation profile"""
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import matplotlib.gridspec as gridspec
        
        # Constants
        mars_radius_km = 3389.5
        deg_to_km_mars = (2 * np.pi * mars_radius_km) / 360
        
        # Load track data
        lat_min, lat_max, lon_min, lon_max = self.define_roi_bounds()
        lat, lon, in_roi, D = self.load_track_data(track_id, lat_min, lat_max, lon_min, lon_max)
        
        if not any(in_roi):
            return
            
        lat_roi = lat[in_roi]
        center_longitude = float(self.config['roi_center'][1])
        lon_roi = center_longitude + (
            (lon[in_roi] - center_longitude + 180) % 360
        ) - 180
        D_clipped = self.preprocess_radargram(D, in_roi, track_id)
        x_km = self.compute_cross_track_axis(D_clipped.shape[1])
        y_microseconds = self.compute_vertical_axis(D_clipped.shape[0])
        
        # Load cluttergram data
        start_index, end_index = self.find_roi_indices(in_roi)
        C = self.process_simulation(track_id, start_index, end_index)
        if C is None:
            print(f"Warning: Could not load cluttergram for track {track_id}")
            C = np.zeros_like(D_clipped)  # Use zeros as fallback
        
        # Filter data for display (70-120 μs)
        valid_rows = (y_microseconds >= 70) & (y_microseconds <= 120)
        D_display = D_clipped[valid_rows, :]
        C_display = C[valid_rows, :]
        y_display = y_microseconds[valid_rows]
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.25, 0.25, 0.25, 0.25], hspace=0.1)
        
        # Subplot 1: Filtered radargram (using Lee filter as example)
        ax1 = plt.subplot(gs[0])
        try:
            D_lee = self.lee_filter(D_display)
            im1 = ax1.imshow(D_lee.T, extent=[y_display.min(), y_display.max(), x_km.min(), x_km.max()], 
                            aspect='auto', cmap='jet', origin='lower')
            ax1.set_ylabel('Distance (km)')
            ax1.set_title(f'Lee Filtered Radargram - {track_id}')
            plt.colorbar(im1, ax=ax1, label='Power (dB)')
        except Exception as e:
            ax1.text(
                0.5, 0.5,
                f'Radargram error: {display_message(str(e))}',
                ha='center', va='center', transform=ax1.transAxes,
            )
        
        # Subplot 2: Cluttergram
        ax2 = plt.subplot(gs[1])
        try:
            im2 = ax2.imshow(C_display.T, extent=[y_display.min(), y_display.max(), x_km.min(), x_km.max()], 
                            aspect='auto', cmap='turbo', origin='lower')
            ax2.set_ylabel('Distance (km)')
            ax2.set_title(f'Cluttergram - {track_id}')
            plt.colorbar(im2, ax=ax2, label='Power (dB)')
        except Exception as e:
            ax2.text(
                0.5, 0.5,
                f'Cluttergram error: {display_message(str(e))}',
                ha='center', va='center', transform=ax2.transAxes,
            )
        
        # Subplot 3: Topographic map
        ax3 = plt.subplot(gs[2])
        self.create_topographic_map_subplot(ax3, track_id, lat_roi, lon_roi)
        ax3.set_title(f'Topographic Map - {track_id}')
        
        # Subplot 4: Elevation profile
        ax4 = plt.subplot(gs[3])
        self.create_elevation_profile_subplot(ax4, track_id, lat_roi, lon_roi)
        ax4.set_title(f'Elevation Profile - {track_id}')
        
        plt.tight_layout()
        output_file = Path(f'./output/maps/{track_id}_joint_visualization.png')
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved joint visualization to {display_path(output_file)}")

    def create_topographic_map(self, track_id):
        """Create a source-specific topographic map for a radar track."""
        if not self.elevation_loaded:
            print(f"Warning: elevation data not available for topographic map of track {track_id}")
            return

        if self.elevation_source == 'mola':
            self.create_mola_topographic_map(track_id)
            return
            
        # Load track data
        lat_min, lat_max, lon_min, lon_max = self.define_roi_bounds()
        lat, lon, in_roi, D = self.load_track_data(track_id, lat_min, lat_max, lon_min, lon_max)
        
        if not any(in_roi):
            print(f"No data in ROI for track {track_id}")
            return
            
        lat_roi = lat[in_roi]
        center_longitude = float(self.config['roi_center'][1])
        lon_roi = center_longitude + (
            (lon[in_roi] - center_longitude + 180) % 360
        ) - 180
        
        # Constants
        mars_radius_km = 3389.5
        deg_to_km_mars = (2 * np.pi * mars_radius_km) / 360
        
        # Convert track coordinates to km
        x_km_hrsc = (lon_roi - lon_roi[0]) * deg_to_km_mars * np.cos(np.radians(lat_roi[0]))
        y_km_hrsc = (lat_roi - lat_roi[0]) * deg_to_km_mars
        
        # Rotate coordinates to align track horizontally
        track_vector = np.array([x_km_hrsc[-1], y_km_hrsc[-1]])
        theta = -np.arctan2(track_vector[1], track_vector[0])
        R_mat = np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta), np.cos(theta)]])
        
        # Calculate track orientation in degrees from north
        # Use the direction from start to end of track in geographic coordinates
        dlon = lon_roi[-1] - lon_roi[0]
        dlat = lat_roi[-1] - lat_roi[0]
        orientation_rad = np.arctan2(dlon, dlat)  # atan2(x, y) gives angle from north
        orientation_deg = (np.degrees(orientation_rad) + 360) % 360  # Ensure 0-360 range
        
        rotated_coords = R_mat @ np.array([x_km_hrsc, y_km_hrsc])
        x_rot = rotated_coords[0]
        y_rot = rotated_coords[1]
        
        # Create grid in rotated coordinates (no buffers to match radargram extent)
        track_length = np.linalg.norm(track_vector)
        # Remove X buffers to start/end exactly at track endpoints
        x_min = x_rot.min()
        x_max = x_rot.max()
        # Dynamic Y buffer based on track length - longer tracks need more vertical coverage
        if track_length > 400:  # Long tracks (>400km) get fixed Y range from -40 to 40
            y_min = -40.0
            y_max = 40.0
        else:  # Short tracks get smaller buffer
            y_buffer = 17.5  # 17.5 km buffer on each side = 35 km total height
            y_min = y_rot.min() - y_buffer
            y_max = y_rot.max() + y_buffer
        
        # Create interpolation grid with reasonable resolution
        # Use a minimum resolution to avoid creating too large grids
        min_res_km = 0.1  # Minimum 100m resolution
        elevation_res_km = max(
            (self.elevation_longitudes[1] - self.elevation_longitudes[0])
            * deg_to_km_mars,
            min_res_km,
        )
        
        # Limit grid size to prevent memory issues
        max_grid_points = 1500  # Maximum 1500x1500 grid (increased for long tracks)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if x_range / elevation_res_km > max_grid_points:
            elevation_res_km = x_range / max_grid_points
        if y_range / elevation_res_km > max_grid_points:
            elevation_res_km = min(elevation_res_km, y_range / max_grid_points)
            
        x_grid = np.arange(x_min, x_max, elevation_res_km)
        y_grid = np.arange(y_min, y_max, elevation_res_km)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Inverse rotation to get back to lat/lon
        inv_R = np.array([[np.cos(theta), np.sin(theta)], 
                         [-np.sin(theta), np.cos(theta)]])
        original_coords = inv_R @ np.array([X_grid.flatten(), Y_grid.flatten()])
        x_orig = original_coords[0] / (deg_to_km_mars * np.cos(np.radians(lat_roi[0]))) + lon_roi[0]
        y_orig = original_coords[1] / deg_to_km_mars + lat_roi[0]
        
        x_orig_grid = x_orig.reshape(X_grid.shape)
        y_orig_grid = y_orig.reshape(Y_grid.shape)
        
        # Interpolate HRSC data using RegularGridInterpolator
        from scipy.interpolate import RegularGridInterpolator
        interp_func = RegularGridInterpolator(
            (self.elevation_latitudes, self.elevation_longitudes),
            self.elevation_data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
        
        # Create points for interpolation
        points = np.column_stack([y_orig_grid.flatten(), x_orig_grid.flatten()])
        elevation_grid = interp_func(points).reshape(X_grid.shape)
        
        # Create plot with poster-optimized size and styling (increased height for better spacing)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(20, 14))
        
        # Poster-optimized fonts
        FONT_SIZE_TITLE = 32
        FONT_SIZE_LABEL = 28
        
        # Plot HRSC elevation
        im = ax.imshow(elevation_grid/1000, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
                      cmap='turbo', origin='lower', aspect='equal')
        
        # Plot track with thicker line for poster visibility
        ax.plot(x_rot, y_rot, 'k-', linewidth=4, label='Radar Track')
        
        # Add horizontal colorbar with poster styling (increased pad for better spacing)
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.12)
        cbar.set_label('Elevation (km)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        cbar.ax.tick_params(labelsize=FONT_SIZE_LABEL)
        
        # Labels and title with poster styling
        ax.set_xlabel('Distance (km)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel('Distance (km)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title(f'HRSC Topographic Map - Track {track_id}\nTrack Orientation: {orientation_deg:.1f}° from North', 
                    fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)
        
        # Set X-axis limits to exactly match track extent (no blank space)
        ax.set_xlim(x_min, x_max)
        
        # Save plot (use lower DPI for large maps to avoid memory issues)
        output_file = Path(self.output_dirs['maps_dir']) / f'{track_id}_topographic_map.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')  # Reduced DPI for large maps
        plt.close()
        
        print(f"Saved topographic map to {display_path(output_file)}")

    def create_mola_topographic_map(self, track_id):
        """Create a MOLA map rotated and clipped to the SHARAD track."""
        lat_min, lat_max, lon_min, lon_max = self.define_roi_bounds()
        lat, lon, in_roi, _ = self.load_track_data(
            track_id, lat_min, lat_max, lon_min, lon_max
        )
        if lat is None or in_roi is None or not np.any(in_roi):
            print(f"No data in ROI for track {track_id}")
            return

        lat_roi = np.asarray(lat[in_roi], dtype=float)
        center_longitude = float(self.config['roi_center'][1])
        lon_roi = center_longitude + (
            (np.asarray(lon[in_roi], dtype=float) - center_longitude + 180) % 360
        ) - 180

        map_data = self.build_track_aligned_mola_grid(lat_roi, lon_roi)
        if map_data is None:
            print(f"No MOLA elevation overlaps track {track_id}")
            return

        x_grid, y_grid, elevation_grid, x_track, y_track, orientation_deg = map_data
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(20, 14))
        image = ax.imshow(
            np.ma.masked_invalid(elevation_grid / 1000),
            extent=[
                x_grid.min(), x_grid.max(),
                y_grid.min(), y_grid.max(),
            ],
            cmap='turbo',
            origin='lower',
            aspect='equal',
        )
        ax.plot(x_track, y_track, 'k-', linewidth=4, label='Radar Track')
        colorbar = fig.colorbar(
            image,
            ax=ax,
            shrink=0.6,
            orientation='horizontal',
            pad=0.12,
        )
        colorbar.set_label('Elevation (km)', fontsize=20, fontweight='bold')
        colorbar.ax.tick_params(labelsize=16)
        ax.set_xlabel('Distance (km)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Distance (km)', fontsize=20, fontweight='bold')
        ax.set_title(
            f'MOLA Topographic Map - Track {track_id}\n'
            f'Track Orientation: {orientation_deg:.1f}° from North',
            fontsize=24,
            fontweight='bold',
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_grid.min(), x_grid.max())
        ax.tick_params(axis='both', which='major', labelsize=16)

        output_file = Path(self.output_dirs['maps_dir']) / f'{track_id}_topographic_map.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved MOLA topographic map to {display_path(output_file)}")

    def build_track_aligned_mola_grid(self, lat_roi, lon_roi):
        """Interpolate MOLA on a clipped grid rotated to follow the track."""
        if len(lat_roi) < 2:
            raise ValueError('At least two track points are required for map rotation')

        mars_radius_km = 3389.5
        degrees_to_km = 2 * np.pi * mars_radius_km / 360
        latitude_reference = float(lat_roi[0])
        delta_longitude = (lon_roi - lon_roi[0] + 180) % 360 - 180
        x_km = (
            delta_longitude
            * degrees_to_km
            * np.cos(np.radians(latitude_reference))
        )
        y_km = (lat_roi - latitude_reference) * degrees_to_km

        track_vector = np.array([x_km[-1], y_km[-1]])
        theta = -np.arctan2(track_vector[1], track_vector[0])
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        x_track, y_track = rotation @ np.array([x_km, y_km])
        track_length = np.linalg.norm(track_vector)
        x_min, x_max = float(x_track.min()), float(x_track.max())
        if track_length > 400:
            y_min, y_max = -40.0, 40.0
        else:
            y_min = float(y_track.min() - 17.5)
            y_max = float(y_track.max() + 17.5)

        resolution_km = max(
            (self.elevation_longitudes[1] - self.elevation_longitudes[0])
            * degrees_to_km
            * np.cos(np.radians(latitude_reference)),
            0.1,
        )
        max_grid_points = 1500
        resolution_km = max(
            resolution_km,
            (x_max - x_min) / max_grid_points,
            (y_max - y_min) / max_grid_points,
        )
        x_grid = np.linspace(
            x_min,
            x_max,
            max(2, int(np.ceil((x_max - x_min) / resolution_km)) + 1),
        )
        y_grid = np.linspace(
            y_min,
            y_max,
            max(2, int(np.ceil((y_max - y_min) / resolution_km)) + 1),
        )
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        inverse_rotation = rotation.T
        original_coordinates = inverse_rotation @ np.array([
            X_grid.ravel(),
            Y_grid.ravel(),
        ])
        latitude_grid = (
            original_coordinates[1] / degrees_to_km + latitude_reference
        )
        longitude_grid = (
            original_coordinates[0]
            / (degrees_to_km * np.cos(np.radians(latitude_reference)))
            + lon_roi[0]
        )
        points = np.column_stack((latitude_grid, longitude_grid))

        from scipy.interpolate import RegularGridInterpolator
        interpolator = RegularGridInterpolator(
            (self.elevation_latitudes, self.elevation_longitudes),
            self.elevation_data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
        elevation_grid = interpolator(points).reshape(X_grid.shape)
        if not np.any(np.isfinite(elevation_grid)):
            return None

        dlon = (lon_roi[-1] - lon_roi[0] + 180) % 360 - 180
        dlat = lat_roi[-1] - lat_roi[0]
        orientation_deg = (
            np.degrees(np.arctan2(dlon, dlat)) + 360
        ) % 360
        return (
            x_grid,
            y_grid,
            elevation_grid,
            x_track,
            y_track,
            orientation_deg,
        )

    def _interpolate_elevation_grid(
        self,
        elevation,
        latitudes,
        longitudes,
    ):
        """Interpolate track samples against a geographic elevation grid."""
        from scipy.interpolate import RegularGridInterpolator

        center_longitude = float(self.config['roi_center'][1])
        longitudes = center_longitude + (
            (np.asarray(longitudes, dtype=float) - center_longitude + 180) % 360
        ) - 180
        interpolator = RegularGridInterpolator(
            (self.elevation_latitudes, self.elevation_longitudes),
            elevation,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
        return interpolator(np.column_stack((latitudes, longitudes)))

    def interpolate_mola_elevation(self, latitudes, longitudes):
        """Sample the loaded MOLA DEM along track."""
        if not self.elevation_loaded or self.elevation_source != 'mola':
            raise RuntimeError('MOLA elevation data is not the selected loaded source')
        return self._interpolate_elevation_grid(
            self.elevation_data,
            latitudes,
            longitudes,
        )

    def interpolate_hrsc_elevation(self, latitudes, longitudes):
        """Sample the loaded HRSC DTM along track."""
        if not self.hrsc_loaded or self.elevation_source != 'hrsc':
            raise RuntimeError('HRSC elevation data is not the selected loaded source')
        return self._interpolate_elevation_grid(
            self.hrsc_elev,
            latitudes,
            longitudes,
        )

    def interpolate_elevation(self, latitudes, longitudes):
        """Sample the configured elevation source along track."""
        if self.elevation_source == 'mola':
            return self.interpolate_mola_elevation(latitudes, longitudes)
        if self.elevation_source == 'hrsc':
            return self.interpolate_hrsc_elevation(latitudes, longitudes)
        raise RuntimeError('No supported elevation source is loaded')

    def create_elevation_profile(self, track_id):
        """Create a profile from the currently configured HRSC or MOLA source."""
        if not self.ensure_elevation_data():
            print(f"Warning: elevation data not available for elevation profile of track {track_id}")
            return
            
        # Load track data
        lat_min, lat_max, lon_min, lon_max = self.define_roi_bounds()
        lat, lon, in_roi, D = self.load_track_data(track_id, lat_min, lat_max, lon_min, lon_max)
        
        if not any(in_roi):
            print(f"No data in ROI for track {track_id}")
            return
            
        lat_roi = lat[in_roi]
        center_longitude = float(self.config['roi_center'][1])
        lon_roi = center_longitude + (
            (lon[in_roi] - center_longitude + 180) % 360
        ) - 180
        # Constants
        mars_radius_km = 3389.5
        deg_to_km_mars = (2 * np.pi * mars_radius_km) / 360
        
        # Account for longitude wrap and the shrinking east-west distance by latitude.
        delta_latitude = np.diff(lat_roi)
        delta_longitude = (
            (np.diff(lon_roi) + 180) % 360
        ) - 180
        mean_latitude = np.radians((lat_roi[1:] + lat_roi[:-1]) / 2)
        segment_distance = deg_to_km_mars * np.sqrt(
            delta_latitude**2
            + (delta_longitude * np.cos(mean_latitude))**2
        )
        dist_km = np.concatenate(([0], np.cumsum(segment_distance)))
        
        # Calculate track orientation in degrees from north
        # Use the direction from start to end of track
        dlon = (lon_roi[-1] - lon_roi[0] + 180) % 360 - 180
        dlat = lat_roi[-1] - lat_roi[0]
        orientation_rad = np.arctan2(dlon, dlat)  # atan2(x, y) gives angle from north
        orientation_deg = (np.degrees(orientation_rad) + 360) % 360  # Ensure 0-360 range
        
        elev_profile = self.interpolate_elevation(lat_roi, lon_roi)
        valid = np.isfinite(elev_profile)
        if not np.any(valid):
            print(
                f"No {self.elevation_source.upper()} elevation samples overlap "
                f"track {track_id}"
            )
            return
        
        # Create plot with poster-optimized styling
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Poster-optimized fonts
        FONT_SIZE_TITLE = 32
        FONT_SIZE_LABEL = 28
        
        # Plot elevation profile with thicker line for poster visibility
        ax.plot(dist_km, elev_profile/1000, 'b-', linewidth=4)
        
        # Labels and title with poster styling
        ax.set_xlabel('Along-track Distance (km)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel('Elevation (km)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title(
                    f'{self.elevation_source.upper()} Elevation Profile - Track {track_id}\n'
                    f'Track Orientation: {orientation_deg:.1f}° from North',
                    fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)
        
        # Set X-axis limits to exactly match radargram extent
        ax.set_xlim(dist_km.min(), dist_km.max())
        
        # Set reasonable y-limits (reduced padding to minimize blank space)
        valid_elevation_km = elev_profile[valid] / 1000
        elev_range = np.ptp(valid_elevation_km)
        if elev_range > 0:
            ax.set_ylim([np.min(valid_elevation_km) - 0.3*elev_range,
                        np.max(valid_elevation_km) + 0.3*elev_range])
        
        # Save plot
        output_file = Path(self.output_dirs['maps_dir']) / f'{track_id}_elevation_profile.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved elevation profile to {display_path(output_file)}")

    def create_topographic_map_subplot(self, ax, track_id, lat_roi, lon_roi):
        """Create topographic map subplot"""
        if not self.elevation_loaded:
            ax.text(0.5, 0.5, 'Elevation data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return

        if self.elevation_source == 'mola':
            try:
                center_longitude = float(self.config['roi_center'][1])
                lon_roi = center_longitude + (
                    (np.asarray(lon_roi) - center_longitude + 180) % 360
                ) - 180
                map_data = self.build_track_aligned_mola_grid(
                    np.asarray(lat_roi),
                    lon_roi,
                )
                if map_data is None:
                    raise ValueError('Track does not overlap the MOLA elevation grid')
                x_grid, y_grid, elevation_grid, x_track, y_track, _ = map_data
                image = ax.imshow(
                    np.ma.masked_invalid(elevation_grid / 1000),
                    extent=[
                        x_grid.min(), x_grid.max(),
                        y_grid.min(), y_grid.max(),
                    ],
                    cmap='turbo',
                    origin='lower',
                    aspect='equal',
                )
                ax.plot(x_track, y_track, 'k-', linewidth=2)
                plt.colorbar(image, ax=ax, label='Elevation (km)')
                ax.set_xlabel('Distance (km)')
                ax.set_ylabel('Distance (km)')
            except Exception as e:
                ax.text(
                    0.5, 0.5,
                    f'MOLA data not available\n{display_message(str(e))}',
                    ha='center', va='center', transform=ax.transAxes,
                )
            return

        try:
            center_longitude = float(self.config['roi_center'][1])
            lon_roi = center_longitude + (
                (np.asarray(lon_roi) - center_longitude + 180) % 360
            ) - 180
            # Use stored HRSC data
            hrsc_elev = self.hrsc_elev
            hrsc_lon_grid = self.hrsc_lon_grid
            hrsc_lat_grid = self.hrsc_lat_grid
            R = self.hrsc_R
            
            # Convert to km coordinates
            mars_radius_km = 3389.5
            deg_to_km_mars = (2 * np.pi * mars_radius_km) / 360
            
            x_km_hrsc = (lon_roi - lon_roi[0]) * deg_to_km_mars * np.cos(np.radians(lat_roi[0]))
            y_km_hrsc = (lat_roi - lat_roi[0]) * deg_to_km_mars
            
            # Rotate to align with track
            track_vector = np.array([x_km_hrsc[-1], y_km_hrsc[-1]])
            theta = -np.arctan2(track_vector[1], track_vector[0])
            R_mat = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta), np.cos(theta)]])
            
            rotated_coords = R_mat @ np.array([x_km_hrsc, y_km_hrsc])
            x_rot = rotated_coords[0]
            y_rot = rotated_coords[1]
            
            # Create grid and interpolate
            track_length = np.linalg.norm(track_vector)
            x_buffer = track_length * 0.1
            y_buffer = track_length * 0.1
            
            x_min, x_max = x_rot.min() - x_buffer, x_rot.max() + x_buffer
            y_min, y_max = y_rot.min() - y_buffer, y_rot.max() + y_buffer
            
            hrsc_res_km = (
                self.elevation_latitudes[1] - self.elevation_latitudes[0]
            ) * deg_to_km_mars
            x_grid = np.arange(x_min, x_max, hrsc_res_km)
            y_grid = np.arange(y_min, y_max, hrsc_res_km)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            
            # Inverse rotation
            inv_R = np.array([[np.cos(theta), np.sin(theta)], 
                             [-np.sin(theta), np.cos(theta)]])
            original_coords = inv_R @ np.array([X_grid.flatten(), Y_grid.flatten()])
            x_orig = original_coords[0] / (deg_to_km_mars * np.cos(np.radians(lat_roi[0]))) + lon_roi[0]
            y_orig = original_coords[1] / deg_to_km_mars + lat_roi[0]
            
            x_orig_grid = x_orig.reshape(X_grid.shape)
            y_orig_grid = y_orig.reshape(Y_grid.shape)
            
            from scipy.interpolate import RegularGridInterpolator
            interp_func = RegularGridInterpolator(
                (self.elevation_latitudes, self.elevation_longitudes),
                self.elevation_data,
                method='linear',
                bounds_error=False,
                fill_value=np.nan,
            )
            
            # Create points for interpolation
            points = np.column_stack([y_orig_grid.flatten(), x_orig_grid.flatten()])
            hrsc_grid = interp_func(points)
            hrsc_grid = hrsc_grid.reshape(X_grid.shape)
            
            # Plot
            im = ax.imshow(hrsc_grid/1000, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
                          cmap='turbo', origin='lower', aspect='equal')
            ax.plot(x_rot, y_rot, 'k-', linewidth=2)
            plt.colorbar(im, ax=ax, label='Elevation (km)')
            ax.set_ylabel('Distance (km)')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'HRSC data not available\n{display_message(str(e))}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    def create_elevation_profile_subplot(self, ax, track_id, lat_roi, lon_roi):
        """Create elevation profile subplot"""
        if not self.ensure_elevation_data():
            ax.text(0.5, 0.5, 'Elevation data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
            
        try:
            # Use stored HRSC data
            # Calculate distance along track
            mars_radius_km = 3389.5
            deg_to_km_mars = (2 * np.pi * mars_radius_km) / 360
            delta_latitude = np.diff(lat_roi)
            delta_longitude = (np.diff(lon_roi) + 180) % 360 - 180
            mean_latitude = np.radians((lat_roi[1:] + lat_roi[:-1]) / 2)
            segment_distance = deg_to_km_mars * np.sqrt(
                delta_latitude**2
                + (delta_longitude * np.cos(mean_latitude))**2
            )
            dist_km = np.concatenate(([0], np.cumsum(segment_distance)))
            elev_profile = self.interpolate_elevation(lat_roi, lon_roi)
            valid = np.isfinite(elev_profile)
            if not np.any(valid):
                raise ValueError('Track does not overlap the elevation grid')
            
            # Plot
            ax.plot(dist_km, elev_profile/1000, 'b-', linewidth=2)
            ax.set_xlabel('Along-track Distance (km)')
            ax.set_ylabel('Elevation (km)')
            ax.set_title(f'{self.elevation_source.upper()} Elevation Profile')
            ax.grid(True)
            
            # Set reasonable y-limits
            valid_elevation_km = elev_profile[valid] / 1000
            elev_range = np.ptp(valid_elevation_km)
            ax.set_ylim([np.min(valid_elevation_km) - 0.6*elev_range,
                        np.max(valid_elevation_km) + 0.6*elev_range])
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Elevation data not available\n{display_message(str(e))}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    def plot_power_vs_time(self, track_id, columns_to_plot=None):
        """Plot power vs time for specified columns"""
        if columns_to_plot is None:
            columns_to_plot = [318, 534, 622]
            
        # Load track data
        lat_min, lat_max, lon_min, lon_max = self.define_roi_bounds()
        lat, lon, in_roi, D = self.load_track_data(track_id, lat_min, lat_max, lon_min, lon_max)
        
        if not any(in_roi):
            return
            
        D_clipped = self.preprocess_radargram(D, in_roi, track_id)
        y_microseconds = self.compute_vertical_axis(D_clipped.shape[0])
        
        # Apply Lee filter
        D_Lee = self.lee_filter(D_clipped)
        
        for col in columns_to_plot:
            if col <= D_Lee.shape[1]:
                self.plot_power_vs_time_single(track_id, y_microseconds, D_Lee[:, col], col, D_Lee)

    def plot_power_vs_time_single(self, track_id, y_microseconds, power_column, col_num, D_Lee):
        """Plot power vs time for a single column with neighbors"""
        import matplotlib.pyplot as plt
        
        # Get neighboring columns
        prev_col = max(1, col_num - 1)
        next_col = min(D_Lee.shape[1], col_num + 1)
        
        # Time mask (75-100 μs)
        time_mask = (y_microseconds >= 75) & (y_microseconds <= 100)
        y_vals = y_microseconds[time_mask]
        
        prev_power = D_Lee[time_mask, prev_col - 1]  # 0-based indexing
        curr_power = power_column[time_mask]
        next_power = D_Lee[time_mask, next_col - 1]  # 0-based indexing
        
        # Create figure with poster styling
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # Poster-optimized fonts
        FONT_SIZE_TITLE = 24
        FONT_SIZE_LABEL = 20
        
        colors = ['blue', 'red', 'green']
        columns = [prev_col, col_num, next_col]
        powers = [prev_power, curr_power, next_power]
        
        for i, (ax, col, power, color) in enumerate(zip(axes, columns, powers, colors)):
            ax.plot(power, y_vals, color=color, linewidth=3)
            ax.set_xlim(-50, -20)
            ax.set_ylim(75, 100)
            ax.set_title(f'Column {col}', fontsize=FONT_SIZE_TITLE, fontweight='bold')
            ax.grid(True, alpha=0.5)
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)
            
            if i == 0:
                ax.set_ylabel('Time Delay (μs)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
            if i == 1:
                ax.set_xlabel('Power (dB)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path(self.output_dirs['plots_dir']) / f'{track_id}_power_cols_{prev_col}-{next_col}.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save CSV data
        import csv
        csv_file = Path(self.output_dirs['plots_dir']) / f'{track_id}_power_data_cols_{prev_col}-{next_col}.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['TimeDelay_us', f'Col{prev_col}_dB', f'Col{col_num}_dB', f'Col{next_col}_dB'])
            for row in zip(y_vals, prev_power, curr_power, next_power):
                writer.writerow(row)

    # ============================================================================
    # MODULAR PROCESSING METHODS - Customizable and Efficient
    # ============================================================================

    def load_elevation_data(self, elevation_file=None, elevation_source='mola'):
        """Load the selected HRSC or MOLA elevation workflow."""
        elevation_source = str(elevation_source).lower()
        if elevation_file is None:
            elevation_file = (
                MOLA_FILE if elevation_source == 'mola'
                else HRSC_DIR / 'dtm_hrsc.tif'
            )
        if elevation_source == 'mola':
            elevation_file = resolve_mola_file(elevation_file)
        self.config['elevation_source'] = elevation_source
        self.config['elevation_file'] = str(elevation_file)
        print(
            f"Loading {elevation_source.upper()} data from "
            f"{display_path(elevation_file)}..."
        )
        start_time = time.time()
        self.clear_elevation_data()

        if elevation_source == 'hrsc':
            elev, lon_grid, lat_grid, R = self.load_hrsc_data(elevation_file)
        elif elevation_source == 'mola':
            elev, lon_grid, lat_grid, R = self.load_mola_data(
                elevation_file,
                self.config['roi_center'],
                self.config['roi_size'],
            )
        else:
            message = f"Unknown elevation source: {elevation_source}"
            logging.error(message)
            print(message)
            return False

        if elev is not None:
            self.set_elevation_data(elev, lon_grid, lat_grid, R, elevation_source)
            print(f"Elevation data loaded in {time.time() - start_time:.2f} seconds")
            return True
        else:
            print(f"Failed to load {elevation_source.upper()} data")
            return False

    def ensure_elevation_data(self):
        """Ensure the configured elevation source, not a different fallback, is loaded."""
        source = str(self.config.get('elevation_source', 'mola')).lower()
        if self.elevation_loaded and self.elevation_source == source:
            return True
        elevation_file = self.config.get('elevation_file') or (
            HRSC_DIR / 'dtm_hrsc.tif' if source == 'hrsc' else MOLA_FILE
        )
        return self.load_elevation_data(elevation_file, source)

    def load_hrsc(self, hrsc_file=HRSC_DIR / 'dtm_hrsc.tif'):
        """Load HRSC elevation data for topographic mapping (backward compatibility)"""
        return self.load_elevation_data(hrsc_file, 'hrsc')

    def process_tracks(self, track_ids=None, generate_radargrams=True, apply_filters=None,
                      generate_maps=True, generate_profiles=True, generate_simulations=True,
                      roi_filter=True, log_gabor_params=None):
        """
        Process multiple tracks with customizable options.

        Parameters:
        - track_ids: List of track IDs to process (None = all available)
        - generate_radargrams: Generate original radargrams
        - apply_filters: List of filters to apply ['gaussian', 'lee', 'log_gabor']
        - generate_maps: Generate topographic maps
        - generate_profiles: Generate elevation profiles
        - generate_simulations: Generate clutter simulations
        - roi_filter: Filter tracks to ROI only
        - log_gabor_params: Dict of parameters for Log-Gabor filter
          {'k': 5, 'nscale': 6, 'mult': 3, 'norient': 6, 'softness': 0.6}
        """
        # Get track list
        if track_ids is None:
            track_ids = self.read_track_ids()

        elevation_ready = True
        if generate_maps or generate_profiles:
            elevation_ready = self.ensure_elevation_data()
            if not elevation_ready:
                print(
                    f"Skipping topographic outputs: configured "
                    f"{self.config.get('elevation_source', 'mola').upper()} data "
                    "could not be loaded."
                )

        processed = 0
        skipped = 0

        for track_id in track_ids:
            try:
                # Check if track is in ROI (if filtering enabled)
                if roi_filter:
                    lat_min, lat_max, lon_min, lon_max = self.define_roi_bounds()
                    lat, lon, in_roi, _ = self.load_track_data(track_id, lat_min, lat_max, lon_min, lon_max)
                    if lat is None or lon is None or in_roi is None:
                        skipped += 1
                        continue
                    if not np.any(in_roi):
                        skipped += 1
                        continue

                # Load radargram data when either its plot or a simulation is requested.
                if generate_radargrams or generate_simulations:
                    self.generate_radargram(
                        track_id,
                        include_simulation=generate_simulations,
                        save_radargram=generate_radargrams
                    )

                # Apply filters
                if apply_filters:
                    self.apply_filters_to_track(track_id, apply_filters, log_gabor_params=log_gabor_params)

                # Generate topographic map
                if generate_maps and elevation_ready:
                    self.generate_topographic_map(track_id)

                # Generate elevation profile
                if generate_profiles and elevation_ready:
                    self.generate_elevation_profile(track_id)

                processed += 1

            except Exception as e:
                # Only log errors, don't print them (progress bar handles user feedback)
                logging.error(
                    f"Error processing track {track_id}: {display_message(str(e))}"
                )
                skipped += 1

        return processed, skipped

    def generate_radargram(self, track_id, include_simulation=True, save_radargram=True):
        """Generate radargram for a single track"""

        # Define ROI bounds
        lat_min, lat_max, lon_min, lon_max = self.define_roi_bounds()

        # Load track data
        lat, lon, in_roi, D = self.load_track_data(track_id, lat_min, lat_max, lon_min, lon_max)
        if lat is None:
            raise ValueError(f"Could not load data for track {track_id}")

        if not np.any(in_roi):
            raise ValueError(f"Track {track_id} has no data in ROI")

        # Preprocess radargram
        D_clipped = self.preprocess_radargram(D, in_roi, track_id)
        x_km = self.compute_cross_track_axis(D_clipped.shape[1])
        y_microseconds = self.compute_vertical_axis(D_clipped.shape[0])

        # Determine ROI indices
        start_index, end_index = self.find_roi_indices(in_roi)
        lat_start, lon_start = lat[start_index], lon[start_index]
        lat_end, lon_end = lat[end_index], lon[end_index]

        if save_radargram:
            self.save_radargram('Radargram', track_id, D_clipped, x_km, y_microseconds,
                                (lat_start, lon_start), (lat_end, lon_end))

        # Process simulation if requested
        if include_simulation:
            try:
                C = self.process_simulation(track_id, start_index, end_index)
                if C is not None:
                    self.save_radargram('Cluttergram', track_id, C, x_km, y_microseconds,
                                        (lat_start, lon_start), (lat_end, lon_end),
                                        directory='simulations')
            except Exception as e:
                logging.warning(
                    f"Simulation failed for {track_id}: {display_message(str(e))}"
                )

        return D_clipped, x_km, y_microseconds, (lat_start, lon_start), (lat_end, lon_end)

    def apply_filters_to_track(self, track_id, filters=['gaussian', 'lee', 'log_gabor'], sigma=1,
                              log_gabor_params=None):
        """
        Apply filters to a track's radargram.

        Parameters:
        - track_id: Track ID to process
        - filters: List of filters ['gaussian', 'lee', 'log_gabor']
        - sigma: Sigma value for Gaussian filter
        - log_gabor_params: Dict of parameters for Log-Gabor filter
          {'k': 5, 'nscale': 6, 'mult': 3, 'norient': 6, 'softness': 0.6}
        """

        # Set default Log-Gabor parameters
        if log_gabor_params is None:
            log_gabor_params = {'k': 5, 'nscale': 6, 'mult': 3, 'norient': 6, 'softness': 0.6}

        # First generate the radargram if it doesn't exist
        try:
            D_clipped, x_km, y_microseconds, start_coords, end_coords = self.generate_radargram(track_id, include_simulation=False)
        except:
            # Try to load existing radargram
            lat_min, lat_max, lon_min, lon_max = self.define_roi_bounds()
            lat, lon, in_roi, D = self.load_track_data(track_id, lat_min, lat_max, lon_min, lon_max)
            D_clipped = self.preprocess_radargram(D, in_roi, track_id)
            x_km = self.compute_cross_track_axis(D_clipped.shape[1])
            y_microseconds = self.compute_vertical_axis(D_clipped.shape[0])
            start_index, end_index = self.find_roi_indices(in_roi)
            start_coords = (lat[start_index], lon[start_index])
            end_coords = (lat[end_index], lon[end_index])

        # Apply requested filters
        if 'gaussian' in filters:
            D_gaussian = ndimage.gaussian_filter(D_clipped, sigma=sigma)
            output_file = self.save_filtered_radargram(track_id, D_gaussian, None, None,
                                                     x_km, y_microseconds, sigma,
                                                     start_coords, end_coords)

        if 'lee' in filters:
            D_lee = self.lee_filter(D_clipped, n=3, ENL=2.5, function_mode='r')
            output_file = self.save_filtered_radargram(track_id, None, D_lee, None,
                                                     x_km, y_microseconds, sigma,
                                                     start_coords, end_coords)

        if 'log_gabor' in filters:
            D_log_gabor = self.noisecomp(D_clipped, **log_gabor_params)
            output_file = self.save_filtered_radargram(track_id, None, None, D_log_gabor,
                                                     x_km, y_microseconds, sigma,
                                                     start_coords, end_coords)

    def generate_topographic_map(self, track_id):
        """Generate topographic map for a single track"""
        if not self.ensure_elevation_data():
            return

        self.create_topographic_map(track_id)

    def generate_elevation_profile(self, track_id):
        """Generate elevation profile for a single track"""
        if not self.ensure_elevation_data():
            return

        self.create_elevation_profile(track_id)

    def batch_process(self, operations, track_ids=None):
        """
        High-level batch processing with operation specifications.

        Example usage:
        processor.batch_process({
            'radargrams': ['01258301', '01823301'],
            'filters': {'tracks': ['01258301'], 'types': ['gaussian', 'lee', 'log_gabor']},
            'maps': ['01258301', '01823301'],
            'profiles': ['01258301']
        })
        """
        print("=== Batch Processing ===")

        # Load the configured source for either map or profile operations.
        needs_elevation = 'maps' in operations or 'profiles' in operations
        elevation_ready = not needs_elevation or self.ensure_elevation_data()

        # Process radargrams
        if 'radargrams' in operations:
            tracks = operations['radargrams']
            print(f"Generating radargrams for {len(tracks)} tracks...")
            for track_id in tracks:
                try:
                    self.generate_radargram(track_id)
                except Exception as e:
                    print(f"Failed radargram for {track_id}: {display_message(str(e))}")

        # Process filters
        if 'filters' in operations:
            filter_config = operations['filters']
            tracks = filter_config.get('tracks', [])
            filter_types = filter_config.get('types', ['gaussian'])
            print(f"Applying {filter_types} filters to {len(tracks)} tracks...")
            for track_id in tracks:
                try:
                    self.apply_filters_to_track(track_id, filter_types)
                except Exception as e:
                    print(f"Failed filters for {track_id}: {display_message(str(e))}")

        # Process maps
        if 'maps' in operations:
            tracks = operations['maps']
            print(f"Generating topographic maps for {len(tracks)} tracks...")
            for track_id in tracks:
                if not elevation_ready:
                    print(f"Skipping map for {track_id}: elevation data unavailable")
                    continue
                try:
                    self.generate_topographic_map(track_id)
                except Exception as e:
                    print(f"Failed map for {track_id}: {display_message(str(e))}")

        # Process profiles
        if 'profiles' in operations:
            tracks = operations['profiles']
            print(f"Generating elevation profiles for {len(tracks)} tracks...")
            for track_id in tracks:
                if not elevation_ready:
                    print(f"Skipping profile for {track_id}: elevation data unavailable")
                    continue
                try:
                    self.generate_elevation_profile(track_id)
                except Exception as e:
                    print(f"Failed profile for {track_id}: {display_message(str(e))}")

        print("=== Batch Processing Complete ===")


# Utility functions for user interface and logging
def setup_logging(log_dir=Path('./output/logs')):
    """Set up logging to both file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'sharad_processing_{timestamp}.log'

    # Configure logging - file gets detailed output, console gets clean output
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)  # Only show errors in console

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )

    return log_file


def print_progress_bar(current, total, prefix='Progress:', suffix='Complete', length=50):
    """
    Print a progress bar to the console.

    Parameters:
    - current: Current progress (int)
    - total: Total items (int)
    - prefix: Text before the progress bar
    - suffix: Text after the progress bar
    - length: Length of the progress bar (int)
    """
    percent = float(current) / total
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)

    # Use carriage return to overwrite the line
    print(f'\r{prefix} |{bar}| {current}/{total} {suffix}', end='', flush=True)

    # Print newline when complete
    if current == total:
        print()


def print_header():
    """Print a nice header for the processing"""
    print("\n" + "="*70)
    print("🚀 SHARAD Mars Radar Processing System")
    print("   Multi-Region Analysis")
    print("="*70)
    print()


def print_summary(stats, start_time, log_file, region_dir):
    """Print a comprehensive summary of all generated products"""
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE")
    print("="*70)
    print(f"⏱️  Total processing time: {duration:.1f} seconds")
    print(
        f"📝 Detailed log saved to: "
        f"./output/{region_dir}/logs/{Path(log_file).name}"
    )
    print()

    print("📊 GENERATED PRODUCTS SUMMARY:")
    print("-" * 40)

    # Radargrams
    if stats['radargrams'] > 0:
        print(f"🖼️  Raw Radargrams:     {stats['radargrams']:2d} files")

    # Simulations
    if stats['simulations'] > 0:
        print(f"🌌 Clutter Simulations: {stats['simulations']:2d} files")

    # Filtered radargrams
    filters_generated = []
    if stats['gaussian'] > 0:
        filters_generated.append(f"Gaussian: {stats['gaussian']}")
    if stats['lee'] > 0:
        filters_generated.append(f"Lee: {stats['lee']}")
    if stats['log_gabor'] > 0:
        filters_generated.append(f"Log-Gabor: {stats['log_gabor']}")

    if filters_generated:
        print(f"🎯 Filtered Radargrams: {stats['total_filtered']:2d} files")
        for filter_info in filters_generated:
            print(f"   • {filter_info}")

    # Maps and profiles
    if stats['maps'] > 0:
        print(f"🗺️  Topographic Maps:   {stats['maps']:2d} files")

    if stats['profiles'] > 0:
        print(f"📈 Elevation Profiles: {stats['profiles']:2d} files")

    print()
    print("📁 Output Locations:")
    print(f"   • Radargrams:   ./output/{region_dir}/radargrams/")
    print(f"   • Filtered:     ./output/{region_dir}/filtered/")
    print(f"   • Maps:         ./output/{region_dir}/maps/")
    print(f"   • Simulations:  ./output/{region_dir}/simulations/")
    print(f"   • Profiles:     ./output/{region_dir}/power_plots/")
    print(f"   • Logs:         ./output/{region_dir}/logs/")
    print()

    # Calculate total files
    total_files = (stats['radargrams'] + stats['simulations'] + stats['total_filtered'] +
                  stats['maps'] + stats['profiles'])

    print(f"📦 Total files generated: {total_files}")
    print("\n🎉 SHARAD processing completed successfully!")
    print("="*70)


def run_batch_processing(config=None):
    """
    Run batch SHARAD processing with customizable configuration and progress display.
    This is the main user-facing function that handles complete processing workflows.

    Parameters:
    - config: Dictionary with processing configuration. If None, uses default Mawrth Vallis config.

    Default config structure:
    {
        'region_name': 'Mawrth Vallis',
        'roi_center': [23.1, 341.2],  # [lat, lon] in degrees
        'roi_size': 7,                # ROI size in degrees
        'elevation_source': 'mola',   # 'mola' or 'hrsc'
        'elevation_file': str(MOLA_FILE),
        'track_file': str(SHARAD_DIR / 'track_ids.txt'),
        'generate_radargrams': True,
        'apply_filters': ['gaussian', 'lee', 'log_gabor'],
        'generate_maps': True,
        'generate_profiles': True,
        'generate_simulations': True,
        'roi_filter': True,
        'log_gabor_params': None
    }
    """

    # Default configuration (Mawrth Vallis)
    if config is None:
        config = {
            'region_name': 'Mawrth Vallis',
            'roi_center': [23.1, 341.2],
            'roi_size': 7,
            'elevation_source': 'mola',
            'elevation_file': str(MOLA_FILE),
            'track_file': str(SHARAD_DIR / 'track_ids.txt'),
            'generate_radargrams': True,
            'apply_filters': ['gaussian', 'lee', 'log_gabor'],
            'generate_maps': True,
            'generate_profiles': True,
            'generate_simulations': True,
            'roi_filter': True,
            'log_gabor_params': None
        }

    region_dir = config['region_name'].lower().replace(' ', '_').replace('/', '_')
    output_root = Path(config.get('output_root', PROJECT_DIR / 'output'))
    region_output_root = output_root / region_dir

    # Setup logging (detailed output goes to file, clean output to console)
    log_file = setup_logging(region_output_root / 'logs')
    logging.info(f"SHARAD {config['region_name']} processing started")

    # Print header
    print_header()

    # Initialize statistics tracking
    stats = {
        'radargrams': 0,
        'simulations': 0,
        'gaussian': 0,
        'lee': 0,
        'log_gabor': 0,
        'total_filtered': 0,
        'maps': 0,
        'profiles': 0,
        'tracks_processed': 0,
        'tracks_skipped': 0
    }

    start_time = time.time()

    try:
        # Initialize the SHARAD processor with custom config
        print("🔧 Initializing SHARAD Processor...")
        processor = SHARADProcessor(create_output_directories=False)

        # Override default config with user config
        processor.config.update({
            'roi_center': config['roi_center'],
            'roi_size': config['roi_size'],
            'datalist_file': config['track_file'],
            'elevation_source': config.get('elevation_source', 'mola'),
            'elevation_file': config.get(
                'elevation_file',
                MOLA_FILE
                if str(config.get('elevation_source', 'mola')).lower() == 'mola'
                else HRSC_DIR / 'dtm_hrsc.tif',
            ),
            'mask_vertical_noise': config.get('mask_vertical_noise', False),
            'noise_channel_min_excess_db': config.get(
                'noise_channel_min_excess_db',
                8.0,
            ),
            'noise_channel_min_persistence': config.get(
                'noise_channel_min_persistence',
                0.7,
            ),
            'noise_channel_width': config.get('noise_channel_width', 1),
        })
        if 'filepath' in config:
            processor.config['filepath'] = config['filepath']

        # Set region-specific output directories
        processor.output_dirs = {
            'root_dir': str(region_output_root),
            'radargrams_dir': str(region_output_root / 'radargrams'),
            'filtered_dir': str(region_output_root / 'filtered'),
            'maps_dir': str(region_output_root / 'maps'),
            'simulations_dir': str(region_output_root / 'simulations'),
            'plots_dir': str(region_output_root / 'power_plots'),
            'logs_dir': str(region_output_root / 'logs')
        }

        # Create the region-specific output directories
        for dir_path in processor.output_dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        logging.info("SHARAD processor initialized")
        print("   ✓ Processor ready")
        print()

        # Load elevation data
        elevation_source = processor.config['elevation_source']
        elevation_file = processor.config['elevation_file']
        if str(elevation_source).lower() == 'mola':
            elevation_file = resolve_mola_file(elevation_file)
            processor.config['elevation_file'] = str(elevation_file)
        
        if os.path.exists(elevation_file):
            print(f"🏔️  Loading {elevation_source.upper()} elevation data...")
            if processor.load_elevation_data(elevation_file, elevation_source):
                logging.info(f"{elevation_source.upper()} elevation data loaded")
            else:
                logging.warning(
                    f"{elevation_source.upper()} elevation data could not be loaded"
                )
        else:
            print(f"⚠️  {elevation_source.upper()} data not found - topographic maps will be skipped")
            logging.warning(
                f"{elevation_source.upper()} file not found: "
                f"{display_path(elevation_file)}"
            )
        print()

        # Get track list
        print("📋 Reading track list...")
        tracks = processor.read_track_ids()
        stats['tracks_processed'] = len(tracks)
        print(f"   📊 Found {len(tracks)} tracks to process")
        logging.info(f"Found {len(tracks)} tracks: {tracks}")
        print()

        # Processing workflow description
        workflow_items = []
        if config['generate_radargrams']:
            workflow_items.append("radargrams")
        if config['apply_filters']:
            workflow_items.append(f"filters ({', '.join(config['apply_filters'])})")
        if config['generate_simulations']:
            workflow_items.append("clutter simulations")
        if config['generate_maps']:
            workflow_items.append("topographic maps")
        if config['generate_profiles']:
            workflow_items.append("elevation profiles")

        print("🔄 Processing Workflow:")
        for item in workflow_items:
            print(f"   • {item}")
        print()

        # Show expected outputs
        expected_outputs = len(tracks) * sum([
            1 if config['generate_radargrams'] else 0,
            len(config['apply_filters']) if config['apply_filters'] else 0,
            1 if config['generate_simulations'] else 0,
            1 if config['generate_maps'] and processor.elevation_loaded else 0,
            1 if config['generate_profiles'] and processor.elevation_loaded else 0
        ])
        print(f"📦 Expected outputs: ~{expected_outputs} files")
        print()

        # Process tracks with progress bar
        print("🚀 Starting batch processing...")
        print()

        # Initialize progress tracking
        processed_count = 0
        total_tracks = len(tracks)

        for i, track_id in enumerate(tracks):
            try:
                # Update progress bar
                print_progress_bar(i, total_tracks,
                                 prefix=f'Processing tracks ({config["region_name"]}):',
                                 suffix=f'Current: {track_id}')

                # Process this track
                processor.process_tracks(
                    track_ids=[track_id],
                    generate_radargrams=config['generate_radargrams'],
                    apply_filters=config['apply_filters'],
                    generate_maps=config['generate_maps'],
                    generate_profiles=config['generate_profiles'],
                    generate_simulations=config['generate_simulations'],
                    roi_filter=config['roi_filter'],
                    log_gabor_params=config['log_gabor_params']
                )

                processed_count += 1

            except Exception as e:
                logging.error(
                    f"Processing failed for track {track_id}: "
                    f"{display_message(str(e))}"
                )
                stats['tracks_skipped'] += 1

        # Final progress bar update
        print_progress_bar(total_tracks, total_tracks,
                         prefix=f'Processing tracks ({config["region_name"]}):',
                         suffix='Complete')

        # Estimate statistics based on successful processing
        successful_tracks = processed_count
        stats['radargrams'] = successful_tracks if config['generate_radargrams'] else 0
        stats['simulations'] = successful_tracks if config['generate_simulations'] else 0

        if config['apply_filters']:
            stats['gaussian'] = successful_tracks if 'gaussian' in config['apply_filters'] else 0
            stats['lee'] = successful_tracks if 'lee' in config['apply_filters'] else 0
            stats['log_gabor'] = successful_tracks if 'log_gabor' in config['apply_filters'] else 0
            stats['total_filtered'] = successful_tracks * len(config['apply_filters'])

        stats['maps'] = successful_tracks if config['generate_maps'] and processor.elevation_loaded else 0
        stats['profiles'] = successful_tracks if config['generate_profiles'] and processor.elevation_loaded else 0

        logging.info("Batch processing completed successfully")

    except Exception as e:
        logging.error(f"Processing failed: {display_message(str(e))}")
        print(
            f"\n❌ Error during processing: {display_message(str(e))}"
        )
        print("Check the log file for details.")
        return

    # Print final summary
    region_dir = config['region_name'].lower().replace(' ', '_').replace('/', '_')
    print_summary(stats, start_time, log_file, region_dir)


# Main execution - Examples of modular usage
if __name__ == "__main__":
    print("=== SHARAD Data Processing Script ===")
    print("Open-source Python implementation for processing Mars SHARAD radar data")
    print(f"Outputs will be saved to {display_path(PROJECT_DIR / 'output')}/")
    print()

    # Initialize processor
    processor = SHARADProcessor()

    print("=== Modular Processing Examples ===")
    print()
    print("1. Load HRSC data once for topographic mapping:")
    print("   processor.load_hrsc('src/hrsc/dtm_hrsc.tif')")
    print()
    print("2. Process specific tracks with custom options:")
    print("   processor.process_tracks(['01258301', '01823301'],")
    print("                          generate_radargrams=True,")
    print("                          apply_filters=['gaussian', 'lee', 'log_gabor'],")
    print("                          generate_maps=True,")
    print("                          generate_profiles=True)")
    print()
    print("3. Generate individual components:")
    print("   processor.generate_radargram('01258301')")
    print("   processor.apply_filters_to_track('01258301', ['gaussian', 'lee', 'log_gabor'])")
    print("   processor.generate_topographic_map('01258301')")
    print("   processor.generate_elevation_profile('01258301')")
    print()
    print("4. Batch processing with operation specifications:")
    print("   processor.batch_process({")
    print("       'radargrams': ['01258301', '01823301'],")
    print("       'filters': {'tracks': ['01258301'], 'types': ['gaussian', 'lee', 'log_gabor']},")
    print("       'maps': ['01258301'],")
    print("       'profiles': ['01258301']")
    print("   })")
    print()

    # Example: Run a quick demo
    print("=== Running Demo Processing ===")
    try:
        # Load HRSC data
        processor.load_hrsc()

        # Process a couple of tracks as demo
        demo_tracks = ['01258301', '01823301']
        processor.process_tracks(demo_tracks[:1],  # Just process first track for demo
                               generate_radargrams=True,
                               apply_filters=['gaussian', 'lee', 'log_gabor'],
                               generate_maps=True,
                               generate_profiles=True)

        print("\n=== Demo Complete ===")
        print("Check ./output/ directory for generated files")
        print("Modify the code above to customize processing for your needs")

    except Exception as e:
        print(f"Demo failed: {display_message(str(e))}")
        print("Find the radargram inputs in src/sharad/radargrams/.")
