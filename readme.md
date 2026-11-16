# SHARAD Processor

This guide covers the standalone SHARAD radar-data processing code in
`sharad_processor.py`. It does not cover the Track Explorer interface, track
discovery, or downloading data.

The processor uses SHARAD radargrams and track geometry to generate radargram
images, filtered images, clutter-simulation comparisons, and (optionally)
terrain maps and elevation profiles. It can be run with the included
`run_processing.py` example or called from another Python script.

## Requirements

Use Python with these packages installed for radargram processing:

```powershell
python -m pip install numpy matplotlib scipy scikit-image
```

Terrain products need additional packages and data:

```powershell
python -m pip install rasterio netCDF4
```

`rasterio` is used for HRSC elevation rasters; `netCDF4` is used for MOLA
NetCDF data. Install only the terrain package needed for your chosen source.

## Input data

By default, the processor reads files from `src/sharad/radargrams/` and looks
for matching clutter simulations in `src/sharad/cluttergrams/`.

| Input | Required for | Files |
| --- | --- | --- |
| Track geometry | All track processing | `s_<track_id>_geom.tab` |
| Radargram measurements and dimensions | All track processing | `s_<track_id>_rgram.img` and `s_<track_id>_rgram.lbl` |
| Clutter simulation | Simulation comparisons | `s_<track_id>_sim.img` and `s_<track_id>_sim.xml` |
| HRSC elevation raster | Maps and profiles using HRSC | A local georeferenced raster, such as a `.tif` |
| MOLA elevation grid | Maps and profiles using MOLA | `src/mola_dem/mola32.nc` (or its matching `.zip` archive) |

Radargram and simulation products are available from the public
[PDS SHARAD USRDR V2 archive](https://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v2/)
and [PDS SHARAD Clutter Simulations archive](https://pds-geosciences.wustl.edu/mro/mro_sharad_simulations/).
The processor does not download these products.

### Track-list format

The first non-empty line of the track file must contain the ROI center latitude,
east longitude, and full width in degrees. Add one track ID per subsequent
line:

```text
23.1,341.2,7
01258301
01823301
```

The ROI is centered on the given latitude and longitude; when ROI filtering is
enabled, each track is clipped to the selected area.

## Run the example batch

The included `run_processing.py` is an editable example configuration. It
reads `src/sharad/mawrth_vallis_tracks.txt` and then calls
`run_batch_processing(config)`.

1. Make sure the input files named above are in their expected folders.
2. Edit the `config` dictionary in `run_processing.py` for your track file,
   region, available elevation data, and desired products.
3. From the project folder, run:

   ```powershell
   python run_processing.py
   ```

Important configuration keys:

| Key | Meaning |
| --- | --- |
| `region_name` | Names the region's output folder. |
| `roi_center`, `roi_size` | ROI center as `[latitude, east_longitude]` and full width in degrees. |
| `track_file` | Track-list path. Its first non-empty line supplies the ROI values. |
| `filepath` | Directory containing radargram and geometry files. Cluttergrams are expected in a sibling `cluttergrams/` folder. |
| `output_root` | Root directory for generated files; defaults to `output/`. |
| `generate_radargrams` | Create original radargram images. |
| `apply_filters` | Any combination of `gaussian`, `lee`, and `log_gabor`; use an empty list to skip filtering. |
| `generate_simulations` | Create cluttergram comparisons when matching simulation products are available. |
| `generate_maps`, `generate_profiles` | Create terrain products using the selected elevation source. |
| `elevation_source`, `elevation_file` | Choose `mola` or `hrsc` and provide its data path. |
| `roi_filter` | Crop processing to the ROI. |

`run_processing.py` also demonstrates optional masking of persistent bright
vertical channels. Set `mask_vertical_noise` to `True` to enable it; the
`noise_channel_*` settings control the detection threshold, persistence, and
neighboring channel width.

## Output

The processor writes files beneath `output/<region_name>/` (the region name is
lowercased and spaces are replaced with underscores):

| Folder | Contents |
| --- | --- |
| `radargrams/` | Original radargram PNGs, named `Radargram_<track_id>.png`. |
| `filtered/` | Filtered radargram PNGs. |
| `simulations/` | Cluttergram comparison PNGs. |
| `maps/` | Topographic maps and elevation profiles. |
| `power_plots/` | Power-vs-time plots and associated data when generated. |
| `logs/` | Run log and `skipped_tracks.txt`. |

Tracks without usable input data or ROI coverage may be skipped. Check the log
and `skipped_tracks.txt` for details.

## Use from Python

For custom workflows, import the batch function and ROI helper:

```python
from sharad_processor import read_roi_from_track_file, run_batch_processing

track_file = "src/sharad/mawrth_vallis_tracks.txt"
roi_center, roi_size = read_roi_from_track_file(track_file)

run_batch_processing({
    "region_name": "Mawrth Vallis",
    "roi_center": roi_center,
    "roi_size": roi_size,
    "track_file": track_file,
    "filepath": "src/sharad/radargrams",
    "output_root": "output",
    "generate_radargrams": True,
    "apply_filters": ["log_gabor"],
    "generate_simulations": False,
    "generate_maps": False,
    "generate_profiles": False,
    "roi_filter": True,
    "log_gabor_params": None,
})
```

For lower-level processing, instantiate `SHARADProcessor` and use methods such
as `process_tracks`, `generate_radargram`, `apply_filters_to_track`,
`generate_topographic_map`, and `generate_elevation_profile`.
