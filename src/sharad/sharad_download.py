"""
This script downloads SHARAD radargrams and cluttergrams from the NASA PDS SHARAD V2 archive and converts geometry
data to GeoJSON for GIS and analysis workflows. This script handles:

1. Downloading radargram data (USRDR V2) from:
   PDS Geosciences Node - SHARAD Radargram Products
   https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/

2. Downloading cluttergram data (surface echo simulations) from:
   PDS Geosciences Node - SHARAD Clutter Simulations
   https://pds-geosciences.wustl.edu/mro/mro_sharad_simulations/

3. Converting geometry files to GeoJSON format for mapping of radar footprints

4. Organizing output files into structured directories for easy access

5. Logging for debugging and tracking progress

- PDS Data Products Guide: 
  https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/mrosh_2101/document/


Usage:
   python sharad_download.py track_ids.txt

   Parameters:
   - track_ids.txt: Text file containing SHARAD track IDs (one per line)
     First line should contain: roi_lat,roi_lon,roi_size
     Subsequent lines contain track IDs

Input:
   Text file containing SHARAD track IDs (one per line), e.g.:
    23.1,341.2,7
    0024_01
    0025_01
    0026_01

Output Structure:
   .
   ├── radargrams/              # Radar observation data
   │   ├── s_*_rgram.img        # Radargram images
   │   ├── s_*_geom.tab         # Geometry tables
   │   └── geojson/             # Converted GeoJSON files
   └── cluttergrams/            # Surface echo simulations
       ├── s_*_sim.img          # Cluttergram images
       └── s_*_browse.tif       # Browse images
"""

import os
import sys
import requests
import certifi
import csv
import geojson
from pathlib import Path
import logging
from typing import Tuple


def get_ca_bundle() -> str:
    """Return the CA bundle for PDS HTTPS requests.

    REQUESTS_CA_BUNDLE supports custom or organization-managed certificates.
    Otherwise, use certifi instead of an unrelated global CURL_CA_BUNDLE setting.
    """
    return os.environ.get('REQUESTS_CA_BUNDLE') or certifi.where()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            Path(__file__).resolve().parent / 'sharad_download.log'
        ),
        logging.StreamHandler()
    ]
)

class RadarDataDownload:
    """Handles all radar download tasks."""
    
    def __init__(self, base_dir: str = '.'):
        """Initialize with base directory.
        
        Args:
            base_dir: Root directory for operations
        """
        self.base_dir = Path(base_dir)
        self.roi_center = None  # Will be set from track file
        self.roi_size = None    # Will be set from track file
        self._setup_directories()
        
    def set_roi(self, lat: float, lon: float, size: float):
        """Set the region of interest for GeoJSON conversion.
        
        Args:
            lat: ROI center latitude in degrees
            lon: ROI center longitude in degrees  
            size: ROI size in degrees
        """
        self.roi_center = (lat, lon)
        self.roi_size = size
        
    def _setup_directories(self) -> None:
        """Create required directory structure."""
        self.dirs = {
            'radargrams': self.base_dir / 'radargrams',
            'cluttergrams': self.base_dir / 'cluttergrams',
            'geojson': self.base_dir / 'radargrams' / 'geojson'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_tracks(self, id_file: str) -> None:
        """download all tracks from the input file.
        
        Args:
            id_file: Path to file containing track IDs
        """
        try:
            with open(id_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error(f"Track ID file not found: {id_file}")
            sys.exit(1)
        
        # Parse ROI parameters from first line
        if not lines:
            logging.error("Track ID file is empty")
            sys.exit(1)
            
        try:
            roi_params = lines[0].split(',')
            if len(roi_params) != 3:
                logging.error("First line must contain: roi_lat,roi_lon,roi_size")
                sys.exit(1)
            roi_lat, roi_lon, roi_size = map(float, roi_params)
            self.set_roi(roi_lat, roi_lon, roi_size)
        except ValueError as e:
            logging.error(f"Invalid ROI parameters in first line: {e}")
            sys.exit(1)
        
        # Get track IDs from remaining lines
        track_ids = lines[1:]
        
        logging.info(f"ROI: lat={roi_lat}, lon={roi_lon}, size={roi_size} degrees")
        logging.info(f"downloading {len(track_ids)} tracksfor the selected region...")
        
        for track_id in track_ids:
            print(f"Downloading track {track_id}...")
            self._download_single_track(track_id)
        
        self._convert_geom_files()
        logging.info("All downloading completed")
        print("Download process completed.")
    
    def _download_single_track(self, track_id: str) -> None:
        """Download radargram and cluttergram files for a single track."""
        logging.info(f"downloading track: {track_id}")
        
        self._download_radargram(track_id)
        self._download_cluttergram(track_id)
    
    def _download_radargram(self, track_id: str) -> None:
        """Download radargram files for a track."""
        base_url = "http://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/mrosh_2101"
        head = track_id[:4]
        
        files = [
            f"{base_url}/data/rgram/s_{head}xx/s_{track_id}_rgram.img",
            f"{base_url}/data/rgram/s_{head}xx/s_{track_id}_rgram.lbl",
            f"{base_url}/data/geom/s_{head}xx/s_{track_id}_geom.lbl",
            f"{base_url}/data/geom/s_{head}xx/s_{track_id}_geom.tab",
            f"{base_url}/browse/tiff/s_{head}xx/s_{track_id}_tiff.lbl",
            f"{base_url}/browse/tiff/s_{head}xx/s_{track_id}_tiff.tif"
        ]
        
        for url in files:
            self._download_file(url, self.dirs['radargrams'])
    
    def _download_cluttergram(self, track_id: str) -> None:
        """Download cluttergram files for a track."""
        base_url = "https://pds-geosciences.wustl.edu/mro/urn-nasa-pds-mro_sharad_simulations"
        head = track_id[:4]
        
        files = [
            f"{base_url}/data/s_{head}xx/s_{track_id}/s_{track_id}_sim.img",
            f"{base_url}/data/s_{head}xx/s_{track_id}/s_{track_id}_sim.xml",
            f"{base_url}/data/s_{head}xx/s_{track_id}/s_{track_id}_emap.img",
            f"{base_url}/data/s_{head}xx/s_{track_id}/s_{track_id}_emap.xml",
            f"{base_url}/data/s_{head}xx/s_{track_id}/s_{track_id}_rtrn.csv",
            f"{base_url}/data/s_{head}xx/s_{track_id}/s_{track_id}_rtrn.xml",
            f"{base_url}/browse/s_{head}xx/s_{track_id}/s_{track_id}_browse_hillshade.tif"
        ]
        
        for url in files:
            self._download_file(url, self.dirs['cluttergrams'])
    
    def _download_file(self, url: str, target_dir: Path) -> None:
        """Download a single file with error handling."""
        filename = url.split('/')[-1]
        target_path = target_dir / filename
        
        if target_path.exists():
            logging.debug(f"File exists, skipping: {filename}")
            return
        
        print(f"Downloading {filename}...")
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=30,
                verify=get_ca_bundle(),
            )
            response.raise_for_status()
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info(f"Downloaded: {filename}")
            print(f"✅ Downloaded {filename}")
        except requests.exceptions.RequestException as e:
            if "404" in str(e):
                logging.info(f"File not available on server: {filename}")
                print(f"⚠️  {filename} not available on server")
            elif isinstance(e, requests.exceptions.SSLError):
                message = (
                    f"TLS certificate verification failed for {filename}. "
                    "Update certifi or set REQUESTS_CA_BUNDLE to a trusted CA file."
                )
                logging.error(f"{message} Details: {e}")
                print(f"❌ {message}")
            else:
                logging.error(f"Failed to download {filename}: {str(e)}")
                print(f"❌ Failed to download {filename}: {str(e)}")
    
    def _convert_geom_files(self) -> None:
        """Convert all geometry files to GeoJSON."""
        logging.info("Converting geometry files to GeoJSON")
        
        for geom_file in self.dirs['radargrams'].glob('*_geom.tab'):
            self._convert_single_geom_file(geom_file)
    
    def _convert_single_geom_file(self, geom_file: Path) -> None:
        """Convert a geometry file to GeoJSON format."""
        output_file = self.dirs['geojson'] / f"{geom_file.stem}.geojson"
        features = []
        
        with open(geom_file, 'r') as infile:
            reader = csv.reader(infile)
            
            for row in reader:
                try:
                    lat, lon = float(row[2]), float(row[3])
                    
                    if self._is_within_roi(lat, lon, self.roi_center, self.roi_size):
                        lon = self._normalize_longitude(lon)
                        
                        feature = geojson.Feature(
                            geometry=geojson.Point((lon, lat)),
                            properties={
                                'track': geom_file.name.split('_')[1],
                                'time': row[1],
                                'mrad': float(row[4]),
                                'srad': float(row[5]),
                                'rvel': float(row[6]),
                                'tvel': float(row[7]),
                                'sza': float(row[8]),
                                'phase': float(row[9])
                            }
                        )
                        features.append(feature)
                except (IndexError, ValueError) as e:
                    logging.warning(f"Skipping malformed row in {geom_file.name}: {str(e)}")
                    continue
        
        if features:
            with open(output_file, 'w') as outfile:
                geojson.dump(geojson.FeatureCollection(features), outfile, indent=2)
            logging.info(f"Created GeoJSON: {output_file.name}")
    
    def check_track_exists(self, track_id: str) -> bool:
        """Check if a track exists in the PDS repository by trying to access its geometry file.
        
        Args:
            track_id: SHARAD track ID
            
        Returns:
            True if track exists, False otherwise
        """
        base_url = "http://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/mrosh_2101"
        head = track_id[:4]
        geom_url = f"{base_url}/data/geom/s_{head}xx/s_{track_id}_geom.lbl"
        
        try:
            response = requests.head(geom_url, timeout=10)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    @staticmethod
    def _is_within_roi(lat: float, lon: float, center: Tuple[float, float], size: float) -> bool:
        """Check if coordinates are within region of interest."""
        lat_min, lat_max = center[0] - size/2, center[0] + size/2
        lon_offset = (lon - center[1] + 180) % 360 - 180
        return lat_min <= lat <= lat_max and abs(lon_offset) <= size/2
    
    @staticmethod
    def _normalize_longitude(lon: float) -> float:
        """Normalize longitude to -180 to 180 range."""
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
        return lon


def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python sharad_download.py track_ids.txt")
        sys.exit(1)
    
    # Parse command line arguments
    track_file = sys.argv[1]
    
    download = RadarDataDownload()
    try:
        download.download_tracks(track_file)
    except Exception as e:
        logging.error(f"downloading failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()