use clap::{Parser, Subcommand, ValueEnum};
use env_logger;
use gdal::{Dataset, DriverManager};
use gdal::raster::Buffer;
use gdal::spatial_ref::{CoordTransform, SpatialRef};
use image::ImageBuffer;
use log::{debug, error, info, warn};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// Coordinate corner reference (for Mode 1)
#[derive(Copy, Clone, Debug, ValueEnum)]
enum CornerRef {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Simple lat/lon struct (for CLI input and metadata)
#[derive(Clone, Debug, Serialize)]
struct LatLon {
    lat: f64,
    lon: f64,
}

impl std::str::FromStr for LatLon {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = s.split(',').map(|p| p.trim()).collect();
        if parts.len() != 2 {
            return Err("Expected 'lat,lon'".into());
        }
        let lat: f64 = parts[0].parse().map_err(|e| format!("lat parse error: {e}"))?;
        let lon: f64 = parts[1].parse().map_err(|e| format!("lon parse error: {e}"))?;
        Ok(LatLon { lat, lon })
    }
}

/// CLI modes
#[derive(Subcommand, Debug)]
enum Mode {
    /// Mode 1: one corner + scale + total size + tile size
    Mode1 {
        /// Input GeoTIFF (DOM/DTM)
        #[arg(long)]
        input: PathBuf,

        /// Corner reference (top-left, top-right, etc.)
        #[arg(long, value_enum)]
        corner_ref: CornerRef,

        /// Corner coordinate (lat,lon) e.g. "59.9127,10.7461"
        #[arg(long)]
        corner_coord: LatLon,

        /// Map scale denominator (e.g. 1000 for 1:1000)
        #[arg(long)]
        scale: f64,

        /// Total physical width in mm
        #[arg(long)]
        width_mm: f64,

        /// Total physical height in mm
        #[arg(long)]
        height_mm: f64,

        /// Tile size in meters (square tiles)
        #[arg(long)]
        tile_size_m: f64,

        /// Output folder for tiles + metadata
        #[arg(long)]
        output_dir: PathBuf,
    },

    /// Mode 2: two opposite corners + (size OR scale) + max tile size
    Mode2 {
        /// Input GeoTIFF (DOM/DTM)
        #[arg(long)]
        input: PathBuf,

        /// First corner (lat,lon)
        #[arg(long)]
        corner_a: LatLon,

        /// Opposite corner (lat,lon)
        #[arg(long)]
        corner_b: LatLon,

        /// Optional total physical width in mm
        #[arg(long)]
        width_mm: Option<f64>,

        /// Optional total physical height in mm
        #[arg(long)]
        height_mm: Option<f64>,

        /// Optional scale denominator (1:scale)
        #[arg(long)]
        scale: Option<f64>,

        /// Max tile size in meters (square tiles)
        #[arg(long)]
        max_tile_size_m: f64,

        /// Output folder for tiles + metadata
        #[arg(long)]
        output_dir: PathBuf,
    },
}

/// Global options
#[derive(Parser, Debug)]
#[command(author, version, about = "GeoTIFF tiler with grid + metadata", long_about = None)]
struct Cli {
    /// Default resolution in meters (ground resolution) – not strictly needed when using geotransform,
    /// but kept as a config knob.
    #[arg(long, default_value_t = 1.0)]
    resolution_m: f64,

    /// Metadata file name (JSON)
    #[arg(long, default_value = "tiles_metadata.json")]
    metadata_file: String,

    /// Enable debug logging
    #[arg(long)]
    debug: bool,

    #[command(subcommand)]
    mode: Mode,
}

/// Grid index for a tile
#[derive(Debug, Clone, Copy, Serialize)]
struct GridIndex {
    row: i32,
    col: i32,
}

/// Metadata for a single tile
#[derive(Debug, Clone, Serialize)]
struct TileMetadata {
    grid: GridIndex,
    top_left: LatLon,
    top_right: LatLon,
    bottom_left: LatLon,
    bottom_right: LatLon,
    width_mm: f64,
    height_mm: f64,
    tile_size_m: f64,
    scale: f64,
    source_file: String,
    output_file: String,
}

/// Metadata for the whole run
#[derive(Debug, Clone, Serialize)]
struct RunMetadata {
    tiles: Vec<TileMetadata>,
    resolution_m: f64,
    grid_origin: LatLon,
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    info!("========== GeoTIFF Tiler Starting ==========");
    let cli = Cli::parse();
    debug!("Parsed CLI args: {:?}", cli);

    if cli.debug {
        warn!("Debug flag is set; for more verbosity, set RUST_LOG=debug");
    }

    info!("Resolution: {:.2}m, Metadata file: {}", cli.resolution_m, cli.metadata_file);

    match cli.mode {
        Mode::Mode1 {
            input,
            corner_ref,
            corner_coord,
            scale,
            width_mm,
            height_mm,
            tile_size_m,
            output_dir,
        } => {
            info!("Running Mode1 with corner ref {:?}", corner_ref);
            if let Err(e) = run_mode1(
                &input,
                corner_ref,
                corner_coord,
                scale,
                width_mm,
                height_mm,
                tile_size_m,
                cli.resolution_m,
                &output_dir,
                &cli.metadata_file,
            ) {
                error!("Mode1 failed: {e}");
                std::process::exit(1);
            } else {
                info!("Mode1 completed successfully");
            }
        }
        Mode::Mode2 {
            input,
            corner_a,
            corner_b,
            width_mm,
            height_mm,
            scale,
            max_tile_size_m,
            output_dir,
        } => {
            info!("Running Mode2 between corners A and B");
            if let Err(e) = run_mode2(
                &input,
                corner_a,
                corner_b,
                width_mm,
                height_mm,
                scale,
                max_tile_size_m,
                cli.resolution_m,
                &output_dir,
                &cli.metadata_file,
            ) {
                error!("Mode2 failed: {e}");
                std::process::exit(1);
            } else {
                info!("Mode2 completed successfully");
            }
        }
    }
    info!("========== GeoTIFF Tiler Complete ==========");
}

fn open_dataset(path: &PathBuf) -> gdal::errors::Result<Dataset> {
    info!("Opening dataset: {}", path.display());
    let ds = Dataset::open(path)?;
    let (w, h) = ds.raster_size();
    info!("Dataset size: {} x {} pixels", w, h);
    
    if let Ok(srs) = ds.spatial_ref() {
        if let Ok(auth) = srs.auth_code() {
            info!("Spatial reference EPSG: {}", auth);
        } else {
            debug!("Spatial reference exists but no EPSG code found");
        }
    } else {
        warn!("No spatial reference found in dataset");
    }
    
    Ok(ds)
}

/// Convert physical size (mm) and scale to ground distance (m)
fn mm_and_scale_to_m(mm: f64, scale: f64) -> f64 {
    let result = (mm / 1000.0) * scale;
    debug!("mm_and_scale_to_m: {}mm at 1:{} = {:.2}m", mm, scale, result);
    result
}

/// Get geotransform and basic checks
fn get_geotransform(ds: &Dataset) -> Result<[f64; 6], String> {
    debug!("Getting geotransform from dataset");
    let gt = ds.geo_transform().map_err(|e| e.to_string())?;
    info!("Geotransform: origin=({:.6}, {:.6}), pixel_size=({:.6}, {:.6})", gt[0], gt[3], gt[1], gt[5]);
    info!("Full geotransform: [{}, {}, {}, {}, {}, {}]", gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]);
    
    if gt[2] != 0.0 || gt[4] != 0.0 {
        warn!("Dataset has rotation (gt[2]={:.6} or gt[4]={:.6} != 0). This code assumes north-up; results may be off.", gt[2], gt[4]);
    }
    
    let (src_w, src_h) = ds.raster_size();
    let (geo_right, geo_bottom) = (
        gt[0] + (src_w as f64) * gt[1],
        gt[3] + (src_h as f64) * gt[5]
    );
    info!("Dataset geo bounds: left={:.6}, top={:.6}, right={:.6}, bottom={:.6}", 
          gt[0], gt[3], geo_right, geo_bottom);
    
    Ok(gt)
}

/// Transform a WGS84 (lat, lon) coordinate into the dataset's CRS.
/// Returns (x, y) in the dataset's native units (e.g. metres for UTM).
fn wgs84_to_dataset_crs(ds: &Dataset, lat: f64, lon: f64) -> Result<(f64, f64), String> {
    let wgs84 = SpatialRef::from_epsg(4326).map_err(|e| e.to_string())?;
    let dst_srs = ds.spatial_ref().map_err(|e| format!("Dataset has no spatial reference: {e}"))?;

    // GDAL axis order: WGS84 is (lat, lon) in GDAL >= 3 unless you force traditional order.
    // We force traditional (lon, lat) order so we can pass x=lon, y=lat directly.
    let mut wgs84_trad = wgs84.clone();
    wgs84_trad.set_axis_mapping_strategy(gdal::spatial_ref::AxisMappingStrategy::TraditionalGisOrder); // OAMS_TRADITIONAL_GIS_ORDER: x=lon, y=lat

    let transform = CoordTransform::new(&wgs84_trad, &dst_srs).map_err(|e| e.to_string())?;

    let mut xs = [lon];
    let mut ys = [lat];
    let mut zs = [0.0_f64];
    transform.transform_coords(&mut xs, &mut ys, &mut zs).map_err(|e| e.to_string())?;

    info!("wgs84_to_dataset_crs: ({:.6}, {:.6}) -> ({:.2}, {:.2})", lat, lon, xs[0], ys[0]);
    Ok((xs[0], ys[0]))
}

/// Convert pixel (x,y) to geo (lon,lat) using geotransform
fn pixel_to_geo(gt: &[f64; 6], px: f64, py: f64) -> (f64, f64) {
    let x = gt[0] + px * gt[1] + py * gt[2];
    let y = gt[3] + px * gt[4] + py * gt[5];
    debug!("pixel_to_geo: pixel ({:.1}, {:.1}) -> geo ({:.6}, {:.6})", px, py, x, y);
    (x, y)
}

/// Convert geo (lon,lat) to pixel (x,y) using geotransform (north-up assumption)
fn geo_to_pixel(gt: &[f64; 6], x: f64, y: f64) -> (f64, f64) {
    let px = (x - gt[0]) / gt[1];
    let py = (y - gt[3]) / gt[5];
    debug!("geo_to_pixel: input geo ({:.6}, {:.6}) using origin ({:.6}, {:.6}) -> pixel ({:.1}, {:.1})", 
           x, y, gt[0], gt[3], px, py);
    
    if px < 0.0 || py < 0.0 {
        debug!("  WARNING: Negative pixel coordinates! Coordinate may be outside dataset bounds (west or north of origin)");
    }
    
    (px, py)
}

/// Compute grid index from pixel offsets and tile size in pixels
fn compute_grid_index_from_pixels(
    origin_px: f64,
    origin_py: f64,
    px: f64,
    py: f64,
    tile_px: f64,
) -> GridIndex {
    let col = ((px - origin_px) / tile_px).floor() as i32;
    let row = ((py - origin_py) / tile_px).floor() as i32;
    debug!("compute_grid_index: origin ({:.1}, {:.1}), pixel ({:.1}, {:.1}), tile_px={:.1} -> grid[{},{}]", 
           origin_px, origin_py, px, py, tile_px, row, col);
    GridIndex { row, col }
}

fn run_mode1(
    input: &PathBuf,
    corner_ref: CornerRef,
    corner_coord: LatLon,
    scale: f64,
    width_mm: f64,
    height_mm: f64,
    tile_size_m: f64,
    resolution_m: f64,
    output_dir: &PathBuf,
    metadata_file: &str,
) -> Result<(), String> {
    info!("\n=== Mode1 Parameters ===");
    info!("  Input: {}", input.display());
    info!("  Corner ref: {:?} at ({:.6}, {:.6})", corner_ref, corner_coord.lat, corner_coord.lon);
    info!("  Scale: 1:{:.0}", scale);
    info!("  Physical size: {:.1}mm x {:.1}mm", width_mm, height_mm);
    info!("  Tile size: {:.2}m", tile_size_m);
    info!("  Output dir: {}", output_dir.display());
    
    debug!("Creating output directory");
    std::fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    let ds = open_dataset(input).map_err(|e| e.to_string())?;
    let gt = get_geotransform(&ds)?;

    let (src_w, src_h) = ds.raster_size();
    let band_count = ds.raster_count();
    info!("Dataset dimensions: {} x {} pixels, {} bands", src_w, src_h, band_count);
    if band_count == 0 {
        error!("Dataset has no bands!");
        return Err("Dataset has no bands".into());
    }

    let pixel_size_x_m = gt[1].abs();
    let pixel_size_y_m = gt[5].abs();

    if pixel_size_x_m <= 0.0 || pixel_size_y_m <= 0.0 {
        error!("Invalid pixel size: {:.4}m x {:.4}m", pixel_size_x_m, pixel_size_y_m);
        return Err(format!("Invalid pixel size: {:.4}m x {:.4}m", pixel_size_x_m, pixel_size_y_m));
    }

    info!(
        "Pixel size ~ {:.4}m x {:.4}m (from geotransform)",
        pixel_size_x_m, pixel_size_y_m
    );

    let total_width_m = mm_and_scale_to_m(width_mm, scale);
    let total_height_m = mm_and_scale_to_m(height_mm, scale);

    info!(
        "Mode1: total area ~ {:.2}m x {:.2}m at 1:{:.0}",
        total_width_m, total_height_m, scale
    );

    // ── FIX: reproject WGS84 input coordinates into the dataset's CRS ──────────
    info!("Input corner from user (WGS84): lat={:.6}, lon={:.6}", corner_coord.lat, corner_coord.lon);
    let (corner_x, corner_y) = wgs84_to_dataset_crs(&ds, corner_coord.lat, corner_coord.lon)?;
    info!("Reprojected to dataset CRS (EPSG:25833): ({:.2}, {:.2})", corner_x, corner_y);
    // ────────────────────────────────────────────────────────────────────────────

    let (area_left, area_right, area_top, area_bottom) = match corner_ref {
        CornerRef::TopLeft     => (corner_x,                corner_x + total_width_m, corner_y,                corner_y - total_height_m),
        CornerRef::TopRight    => (corner_x - total_width_m, corner_x,               corner_y,                corner_y - total_height_m),
        CornerRef::BottomLeft  => (corner_x,                corner_x + total_width_m, corner_y + total_height_m, corner_y),
        CornerRef::BottomRight => (corner_x - total_width_m, corner_x,               corner_y + total_height_m, corner_y),
    };
    
    info!(
        "Corner {:?} at CRS ({:.2},{:.2}), area bounds: X=[{:.2}, {:.2}], Y=[{:.2}, {:.2}]",
        corner_ref, corner_x, corner_y, area_left, area_right, area_bottom, area_top
    );

    let (area_left_px_f, area_top_px_f) = geo_to_pixel(&gt, area_left, area_top);
    let (area_right_px_f, area_bottom_px_f) = geo_to_pixel(&gt, area_right, area_bottom);
    let area_left_px = area_left_px_f.round() as isize;
    let area_top_px = area_top_px_f.round() as isize;
    let area_right_px = area_right_px_f.round() as isize;
    let area_bottom_px = area_bottom_px_f.round() as isize;

    let area_top_left_px = area_left_px.min(area_right_px);
    let area_top_left_py = area_top_px.min(area_bottom_px);
    let area_width_px = (area_right_px - area_left_px).abs();
    let area_height_px = (area_bottom_px - area_top_px).abs();

    info!("Dataset pixel bounds: 0 to {} (x), 0 to {} (y)", src_w as i32 - 1, src_h as i32 - 1);
    info!(
        "Area bounds in pixels: top-left ({}, {}), size: {} x {} pixels",
        area_top_left_px, area_top_left_py, area_width_px, area_height_px
    );
    
    if area_top_left_px < 0 || area_top_left_py < 0 {
        warn!("Computed area top-left ({},{}) is outside dataset - may have tiles outside bounds", area_top_left_px, area_top_left_py);
    }

    let tile_px = (tile_size_m / pixel_size_x_m).round().max(1.0) as isize;
    let tiles_x = ((area_width_px as f64) / (tile_px as f64)).ceil() as i32;
    let tiles_y = ((area_height_px as f64) / (tile_px as f64)).ceil() as i32;
    let total_tiles = (tiles_x as u32) * (tiles_y as u32);

    info!(
        "Tiles: {} (x) x {} (y), tile_size_m ~ {:.2}, tile_px ~ {} (total {} tiles)",
        tiles_x, tiles_y, tile_size_m, tile_px, total_tiles
    );

    let (grid_origin_x, grid_origin_y) =
        pixel_to_geo(&gt, area_top_left_px as f64, area_top_left_py as f64);
    let grid_origin = LatLon {
        lat: grid_origin_y,
        lon: grid_origin_x,
    };

    let driver = DriverManager::get_driver_by_name("GTiff").map_err(|e| e.to_string())?;
    let mut tiles_meta = Vec::new();
    let mut processed_tiles = 0;
    let mut skipped_tiles = 0;

    info!("Starting tile generation for {} x {} grid", tiles_x, tiles_y);
    for row in 0..tiles_y {
        for col in 0..tiles_x {
            let tile_left   = area_left + (col as f64) * tile_size_m;
            let tile_right  = tile_left + tile_size_m;
            let tile_top    = area_top  - (row as f64) * tile_size_m;
            let tile_bottom = tile_top  - tile_size_m;
            
            let (tile_left_px_f, tile_top_px_f) = geo_to_pixel(&gt, tile_left, tile_top);
            let (_tile_right_px_f, tile_bottom_px_f) = geo_to_pixel(&gt, tile_right, tile_bottom);
            
            let x_off = tile_left_px_f.round() as isize;
            let y_off = tile_top_px_f.min(tile_bottom_px_f).round() as isize;

            if x_off >= src_w as isize || y_off >= src_h as isize {
                debug!("Tile [{}][{}] at pixel ({},{}) is outside dataset bounds, skipping", row, col, x_off, y_off);
                skipped_tiles += 1;
                continue;
            }
            if x_off < 0 || y_off < 0 {
                debug!("Tile [{}][{}] at pixel ({},{}) starts before dataset origin, skipping", row, col, x_off, y_off);
                skipped_tiles += 1;
                continue;
            }

            let mut w = tile_px;
            let mut h = tile_px;
            if x_off + w > src_w as isize { w = src_w as isize - x_off; }
            if y_off + h > src_h as isize { h = src_h as isize - y_off; }

            if w <= 0 || h <= 0 {
                debug!("Tile [{}][{}] has invalid dimensions {}x{}, skipping", row, col, w, h);
                skipped_tiles += 1;
                continue;
            }

            let (tl_x, tl_y) = pixel_to_geo(&gt, x_off as f64, y_off as f64);
            let (tr_x, tr_y) = pixel_to_geo(&gt, (x_off + w) as f64, y_off as f64);
            let (bl_x, bl_y) = pixel_to_geo(&gt, x_off as f64, (y_off + h) as f64);
            let (br_x, br_y) = pixel_to_geo(&gt, (x_off + w) as f64, (y_off + h) as f64);

            let top_left    = LatLon { lat: tl_y, lon: tl_x };
            let top_right   = LatLon { lat: tr_y, lon: tr_x };
            let bottom_left = LatLon { lat: bl_y, lon: bl_x };
            let bottom_right= LatLon { lat: br_y, lon: br_x };

            let grid = compute_grid_index_from_pixels(
                area_top_left_px as f64, area_top_left_py as f64,
                x_off as f64, y_off as f64, tile_px as f64,
            );

            let out_name = format!("tile_r{}_c{}.tif", row, col);
            let out_path = output_dir.join(&out_name);
            info!(
                "Writing tile {} at pixel window x_off={}, y_off={}, w={}, h={}",
                out_name, x_off, y_off, w, h
            );

            let mut out_ds = driver
                .create_with_band_type::<f32, _>(
                    out_path.to_str().unwrap(),
                    w as usize, h as usize, band_count as usize,
                )
                .map_err(|e| e.to_string())?;

            let mut tile_gt = gt;
            tile_gt[0] = tl_x;
            tile_gt[3] = tl_y;
            out_ds.set_geo_transform(&tile_gt).map_err(|e| e.to_string())?;
            if let Ok(srs) = ds.spatial_ref() {
                out_ds.set_spatial_ref(&srs).map_err(|e| e.to_string())?;
            }

            for b in 1..=band_count {
                debug!("Processing band {} of {}", b, band_count);
                let src_band = ds.rasterband(b).map_err(|e| e.to_string())?;
                let mut dst_band = out_ds.rasterband(b).map_err(|e| e.to_string())?;

                let mut buffer: Buffer<f32> = src_band
                    .read_as(
                        (x_off as isize, y_off as isize),
                        (w as usize, h as usize),
                        (w as usize, h as usize),
                        None,
                    )
                    .map_err(|e| e.to_string())?;

                dst_band
                    .write((0, 0), (w as usize, h as usize), &mut buffer)
                    .map_err(|e| e.to_string())?;
            }
            processed_tiles += 1;

            let tile_meta = TileMetadata {
                grid,
                top_left, top_right, bottom_left, bottom_right,
                width_mm: width_mm / tiles_x as f64,
                height_mm: height_mm / tiles_y as f64,
                tile_size_m, scale,
                source_file: input.display().to_string(),
                output_file: out_path.display().to_string(),
            };
            debug!("Tile meta: {:?}", tile_meta);
            tiles_meta.push(tile_meta);
        }
    }

    info!("Tile generation complete: {} processed, {} skipped", processed_tiles, skipped_tiles);
    let run_meta = RunMetadata { tiles: tiles_meta.clone(), resolution_m, grid_origin };
    info!("Generated {} tiles total (in metadata)", tiles_meta.len());
    write_metadata(output_dir, metadata_file, &run_meta)?;
    
    // Generate PNG preview of the selected area
    generate_area_preview(&ds, &gt, area_top_left_px, area_top_left_py, area_width_px, area_height_px, output_dir)?;
    
    // Generate overlay showing selection and tiles on original data
    generate_selection_overlay(&ds, area_top_left_px, area_top_left_py, area_width_px, area_height_px, 
                             tile_px, tiles_x, tiles_y, output_dir)?;
    
    info!("Mode1 finished successfully");
    Ok(())
}

fn run_mode2(
    input: &PathBuf,
    corner_a: LatLon,
    corner_b: LatLon,
    width_mm: Option<f64>,
    height_mm: Option<f64>,
    scale: Option<f64>,
    max_tile_size_m: f64,
    resolution_m: f64,
    output_dir: &PathBuf,
    metadata_file: &str,
) -> Result<(), String> {
    info!("\n=== Mode2 Parameters ===");
    info!("  Input: {}", input.display());
    info!("  Corner A: ({:.6}, {:.6})", corner_a.lat, corner_a.lon);
    info!("  Corner B: ({:.6}, {:.6})", corner_b.lat, corner_b.lon);
    info!("  Max tile size: {:.2}m", max_tile_size_m);
    info!("  Width: {:?}mm, Height: {:?}mm, Scale: {:?}", width_mm, height_mm, scale);
    info!("  Output dir: {}", output_dir.display());
    
    std::fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    let ds = open_dataset(input).map_err(|e| e.to_string())?;
    let gt = get_geotransform(&ds)?;

    let (src_w, src_h) = ds.raster_size();
    let band_count = ds.raster_count();
    info!("Dataset dimensions: {} x {} pixels, {} bands", src_w, src_h, band_count);
    if band_count == 0 {
        return Err("Dataset has no bands".into());
    }
    let _band1 = ds.rasterband(1).map_err(|e| e.to_string())?;

    let pixel_size_x_m = gt[1].abs();
    let pixel_size_y_m = gt[5].abs();
    if pixel_size_x_m <= 0.0 || pixel_size_y_m <= 0.0 {
        return Err(format!("Invalid pixel size: {:.4}m x {:.4}m", pixel_size_x_m, pixel_size_y_m));
    }
    info!("Pixel size: {:.4}m x {:.4}m (from geotransform)", pixel_size_x_m, pixel_size_y_m);

    // ── FIX: reproject WGS84 input coordinates into the dataset's CRS ──────────
    info!("Input corners (WGS84): A=({:.6},{:.6}) B=({:.6},{:.6})",
          corner_a.lat, corner_a.lon, corner_b.lat, corner_b.lon);
    let (ax_transformed, ay_transformed) = wgs84_to_dataset_crs(&ds, corner_a.lat, corner_a.lon)?;
    let (bx_transformed, by_transformed) = wgs84_to_dataset_crs(&ds, corner_b.lat, corner_b.lon)?;
    info!("Reprojected to dataset CRS:");
    info!("  Corner A: ({:.2}, {:.2})", ax_transformed, ay_transformed);
    info!("  Corner B: ({:.2}, {:.2})", bx_transformed, by_transformed);
    // ────────────────────────────────────────────────────────────────────────────

    let (apx_f, apy_f) = geo_to_pixel(&gt, ax_transformed, ay_transformed);
    let (bpx_f, bpy_f) = geo_to_pixel(&gt, bx_transformed, by_transformed);
    info!("Pixel coordinates: A=({:.1},{:.1}) B=({:.1},{:.1})", apx_f, apy_f, bpx_f, bpy_f);
    info!("Dataset pixel bounds: 0 to {} (x), 0 to {} (y)", src_w as i32 - 1, src_h as i32 - 1);

    if apx_f < 0.0 || apx_f > src_w as f64 || apy_f < 0.0 || apy_f > src_h as f64 {
        warn!("Corner A pixel ({:.1}, {:.1}) is OUTSIDE dataset bounds!", apx_f, apy_f);
    }
    if bpx_f < 0.0 || bpx_f > src_w as f64 || bpy_f < 0.0 || bpy_f > src_h as f64 {
        warn!("Corner B pixel ({:.1}, {:.1}) is OUTSIDE dataset bounds!", bpx_f, bpy_f);
    }

    let min_px = apx_f.min(bpx_f);
    let max_px = apx_f.max(bpx_f);
    let min_py = apy_f.min(bpy_f);
    let max_py = apy_f.max(bpy_f);

    let total_px_x = (max_px - min_px).abs().round() as isize;
    let total_px_y = (max_py - min_py).abs().round() as isize;

    let total_width_m  = total_px_x as f64 * pixel_size_x_m;
    let total_height_m = total_px_y as f64 * pixel_size_y_m;

    info!("Mode2: ground extent ~ {:.2}m x {:.2}m between corners", total_width_m, total_height_m);

    let (scale_final, width_mm_final, height_mm_final) = match (scale, width_mm, height_mm) {
        (Some(s), Some(wmm), Some(hmm)) => (s, wmm, hmm),
        (Some(s), None, None) => {
            let wmm = (total_width_m / s) * 1000.0;
            let hmm = (total_height_m / s) * 1000.0;
            (s, wmm, hmm)
        }
        (None, Some(wmm), Some(hmm)) => {
            let s = total_width_m / (wmm / 1000.0);
            (s, wmm, hmm)
        }
        _ => return Err(
            "Provide either: (scale + width_mm + height_mm) OR (scale only) OR (width_mm + height_mm)".into()
        ),
    };

    info!("Mode2: using scale 1:{:.0}, physical size ~ {:.1}mm x {:.1}mm",
          scale_final, width_mm_final, height_mm_final);

    let tile_px_max = (max_tile_size_m / pixel_size_x_m).round().max(1.0) as isize;
    let tiles_x = ((total_px_x as f64) / (tile_px_max as f64)).ceil().max(1.0) as i32;
    let tiles_y = ((total_px_y as f64) / (tile_px_max as f64)).ceil().max(1.0) as i32;

    let tile_px_x = (total_px_x as f64 / tiles_x as f64).ceil() as isize;
    let tile_px_y = (total_px_y as f64 / tiles_y as f64).ceil() as isize;
    let tile_px = tile_px_x.min(tile_px_y);

    let tile_size_m = tile_px as f64 * pixel_size_x_m;
    let total_tiles  = (tiles_x as u32) * (tiles_y as u32);

    info!("Tiles: {} (x) x {} (y), tile_size_m ~ {:.2}, tile_px ~ {} (total {} tiles)",
          tiles_x, tiles_y, tile_size_m, tile_px, total_tiles);

    let area_top_left_px = min_px.round() as isize;
    let area_top_left_py = min_py.round() as isize;

    let (grid_origin_x, grid_origin_y) =
        pixel_to_geo(&gt, area_top_left_px as f64, area_top_left_py as f64);
    let grid_origin = LatLon { lat: grid_origin_y, lon: grid_origin_x };

    let driver = DriverManager::get_driver_by_name("GTiff").map_err(|e| e.to_string())?;
    let mut tiles_meta = Vec::new();
    let mut processed_tiles = 0;
    let mut skipped_tiles  = 0;

    info!("Starting tile generation for {} x {} grid", tiles_x, tiles_y);
    for row in 0..tiles_y {
        for col in 0..tiles_x {
            let x_off = area_top_left_px + (col as isize) * tile_px;
            let y_off = area_top_left_py + (row as isize) * tile_px;

            if x_off >= src_w as isize || y_off >= src_h as isize || x_off < 0 || y_off < 0 {
                skipped_tiles += 1;
                continue;
            }

            let mut w = tile_px;
            let mut h = tile_px;
            if x_off + w > src_w as isize { w = src_w as isize - x_off; }
            if y_off + h > src_h as isize { h = src_h as isize - y_off; }

            if w <= 0 || h <= 0 { skipped_tiles += 1; continue; }

            let (tl_x, tl_y) = pixel_to_geo(&gt, x_off as f64, y_off as f64);
            let (tr_x, tr_y) = pixel_to_geo(&gt, (x_off + w) as f64, y_off as f64);
            let (bl_x, bl_y) = pixel_to_geo(&gt, x_off as f64, (y_off + h) as f64);
            let (br_x, br_y) = pixel_to_geo(&gt, (x_off + w) as f64, (y_off + h) as f64);

            let top_left    = LatLon { lat: tl_y, lon: tl_x };
            let top_right   = LatLon { lat: tr_y, lon: tr_x };
            let bottom_left = LatLon { lat: bl_y, lon: bl_x };
            let bottom_right= LatLon { lat: br_y, lon: br_x };

            let grid = compute_grid_index_from_pixels(
                area_top_left_px as f64, area_top_left_py as f64,
                x_off as f64, y_off as f64, tile_px as f64,
            );

            let out_name = format!("tile_r{}_c{}.tif", row, col);
            let out_path = output_dir.join(&out_name);
            info!("Writing tile {} at pixel window x_off={}, y_off={}, w={}, h={}", out_name, x_off, y_off, w, h);

            let mut out_ds = driver
                .create_with_band_type::<f32, _>(
                    out_path.to_str().unwrap(),
                    w as usize, h as usize, band_count as usize,
                )
                .map_err(|e| e.to_string())?;

            let mut tile_gt = gt;
            tile_gt[0] = tl_x;
            tile_gt[3] = tl_y;
            out_ds.set_geo_transform(&tile_gt).map_err(|e| e.to_string())?;
            if let Ok(srs) = ds.spatial_ref() {
                out_ds.set_spatial_ref(&srs).map_err(|e| e.to_string())?;
            } else {
                warn!("No spatial reference found in source dataset");
            }

            for b in 1..=band_count {
                let src_band = ds.rasterband(b).map_err(|e| e.to_string())?;
                let mut dst_band = out_ds.rasterband(b).map_err(|e| e.to_string())?;

                let mut buffer: Buffer<f32> = src_band
                    .read_as(
                        (x_off as isize, y_off as isize),
                        (w as usize, h as usize),
                        (w as usize, h as usize),
                        None,
                    )
                    .map_err(|e| e.to_string())?;

                dst_band
                    .write((0, 0), (w as usize, h as usize), &mut buffer)
                    .map_err(|e| e.to_string())?;
            }
            processed_tiles += 1;

            let tile_meta = TileMetadata {
                grid,
                top_left, top_right, bottom_left, bottom_right,
                width_mm: width_mm_final / tiles_x as f64,
                height_mm: height_mm_final / tiles_y as f64,
                tile_size_m, scale: scale_final,
                source_file: input.display().to_string(),
                output_file: out_path.display().to_string(),
            };
            tiles_meta.push(tile_meta);
        }
    }

    info!("Tile generation complete: {} processed, {} skipped", processed_tiles, skipped_tiles);
    let run_meta = RunMetadata { tiles: tiles_meta.clone(), resolution_m, grid_origin };
    info!("Generated {} tiles total (in metadata)", tiles_meta.len());
    write_metadata(output_dir, metadata_file, &run_meta)?;
    
    // Generate PNG preview of the selected area
    generate_area_preview(&ds, &gt, area_top_left_px, area_top_left_py, total_px_x, total_px_y, output_dir)?;
    
    // Generate overlay showing selection and tiles on original data
    generate_selection_overlay(&ds, area_top_left_px, area_top_left_py, total_px_x, total_px_y,
                             tile_px, tiles_x, tiles_y, output_dir)?;
    
    info!("Mode2 finished successfully");
    Ok(())
}

fn write_metadata(
    output_dir: &PathBuf,
    metadata_file: &str,
    run_meta: &RunMetadata,
) -> Result<(), String> {
    std::fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    let path = output_dir.join(metadata_file);
    info!("Writing metadata to {}", path.display());

    let mut f = File::create(&path).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(run_meta).map_err(|e| e.to_string())?;
    f.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
    info!("Metadata written successfully");
    Ok(())
}

/// Generate a PNG preview of the selected area from the dataset
fn generate_area_preview(
    ds: &Dataset,
    _gt: &[f64; 6],
    x_off: isize,
    y_off: isize,
    width: isize,
    height: isize,
    output_dir: &PathBuf,
) -> Result<(), String> {
    info!("Generating PNG preview of selected area with hillshade");
    
    let (src_w, src_h) = ds.raster_size();
    
    // Clamp the area to dataset bounds
    let x_start = x_off.max(0).min(src_w as isize - 1) as isize;
    let y_start = y_off.max(0).min(src_h as isize - 1) as isize;
    let x_end = (x_off + width).min(src_w as isize).max(0) as isize;
    let y_end = (y_off + height).min(src_h as isize).max(0) as isize;
    
    let preview_width = (x_end - x_start).max(1) as usize;
    let preview_height = (y_end - y_start).max(1) as usize;
    
    // Scale down if too large for preview
    let (display_width, display_height) = if preview_width > 2000 || preview_height > 2000 {
        let scale = ((preview_width as f64).max(preview_height as f64) / 2000.0).ceil() as usize;
        (preview_width / scale, preview_height / scale)
    } else {
        (preview_width, preview_height)
    };
    
    info!("Reading preview area: {:.0} x {:.0} pixels at offset ({}, {})", 
          display_width, display_height, x_start, y_start);
    
    // Read the first band as elevation/terrain data
    let band1 = ds.rasterband(1).map_err(|e| e.to_string())?;
    let buffer: Buffer<f32> = band1
        .read_as(
            (x_start, y_start),
            (preview_width, preview_height),
            (display_width, display_height),
            None,
        )
        .map_err(|e| e.to_string())?;
    
    let data = buffer.data();
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = if (max_val - min_val).abs() > 0.001 {
        max_val - min_val
    } else {
        1.0
    };
    
    info!("Data range: {:.2} to {:.2}", min_val, max_val);
    
    // Compute hillshade
    let mut img = ImageBuffer::new(display_width as u32, display_height as u32);
    
    // Hillshade parameters
    let azimuth = 315.0_f32.to_radians(); // Light from upper left
    let altitude = 45.0_f32.to_radians(); // 45 degree angle
    let z_factor = 10.0_f32; // Vertical exaggeration
    
    // Store shade values for normalization
    let mut shades = vec![0.0_f32; display_width * display_height];
    let mut min_shade = f32::INFINITY;
    let mut max_shade = f32::NEG_INFINITY;
    
    for y in 0..display_height {
        for x in 0..display_width {
            let idx = y * display_width + x;
            
            // Get center elevation (normalized)
            let center = if idx < data.len() {
                (data[idx] - min_val) / range
            } else {
                0.0
            };
            
            // Get neighboring elevations using Sobel operators
            let tl = if x > 0 && y > 0 && (y - 1) * display_width + (x - 1) < data.len() {
                (data[(y - 1) * display_width + (x - 1)] - min_val) / range
            } else {
                center
            };
            
            let tr = if x < display_width - 1 && y > 0 && (y - 1) * display_width + (x + 1) < data.len() {
                (data[(y - 1) * display_width + (x + 1)] - min_val) / range
            } else {
                center
            };
            
            let bl = if x > 0 && y < display_height - 1 && (y + 1) * display_width + (x - 1) < data.len() {
                (data[(y + 1) * display_width + (x - 1)] - min_val) / range
            } else {
                center
            };
            
            let br = if x < display_width - 1 && y < display_height - 1 && (y + 1) * display_width + (x + 1) < data.len() {
                (data[(y + 1) * display_width + (x + 1)] - min_val) / range
            } else {
                center
            };
            
            let left = if x > 0 && y * display_width + (x - 1) < data.len() {
                (data[y * display_width + (x - 1)] - min_val) / range
            } else {
                center
            };
            
            let right = if x < display_width - 1 && y * display_width + (x + 1) < data.len() {
                (data[y * display_width + (x + 1)] - min_val) / range
            } else {
                center
            };
            
            let top = if y > 0 && (y - 1) * display_width + x < data.len() {
                (data[(y - 1) * display_width + x] - min_val) / range
            } else {
                center
            };
            
            let bottom = if y < display_height - 1 && (y + 1) * display_width + x < data.len() {
                (data[(y + 1) * display_width + x] - min_val) / range
            } else {
                center
            };
            
            // Sobel operators for better gradients
            let dx = (-tl + tr - 2.0 * left + 2.0 * right - bl + br) * z_factor / 8.0;
            let dy = (-tl - 2.0 * top - tr + bl + 2.0 * bottom + br) * z_factor / 8.0;
            
            // Compute surface normal
            let nx = -dx;
            let ny = -dy;
            let nz = 1.0_f32;
            
            let norm = (nx * nx + ny * ny + nz * nz).sqrt();
            let nx = nx / norm;
            let ny = ny / norm;
            let nz = nz / norm;
            
            // Light direction (from altitude and azimuth)
            let lx = altitude.cos() * azimuth.sin();
            let ly = altitude.cos() * azimuth.cos();
            let lz = altitude.sin();
            
            // Compute shading (dot product of normal and light direction)
            let shade = nx * lx + ny * ly + nz * lz;
            
            shades[idx] = shade;
            min_shade = min_shade.min(shade);
            max_shade = max_shade.max(shade);
        }
    }
    
    // Normalize shades and render
    let shade_range = if (max_shade - min_shade).abs() > 0.001 {
        max_shade - min_shade
    } else {
        1.0
    };
    
    info!("Shade range: {:.4} to {:.4}", min_shade, max_shade);
    
    for y in 0..display_height {
        for x in 0..display_width {
            let idx = y * display_width + x;
            let normalized_shade = (shades[idx] - min_shade) / shade_range;
            // Apply contrast boost using power function
            let boosted = normalized_shade.powf(0.7);
            let gray = ((boosted * 255.0).clamp(0.0, 255.0)) as u8;
            *img.get_pixel_mut(x as u32, y as u32) = image::Luma([gray]);
        }
    }
    
    // Save PNG
    let preview_path = output_dir.join("area_preview.png");
    img.save(&preview_path).map_err(|e| e.to_string())?;
    info!("PNG hillshade preview saved to {}", preview_path.display());
    
    Ok(())
}

/// Generate a PNG showing the original data with the selected area marked as a red rectangle and tiles as white grid
fn generate_selection_overlay(
    ds: &Dataset,
    x_off: isize,
    y_off: isize,
    width: isize,
    height: isize,
    tile_px: isize,
    tiles_x: i32,
    tiles_y: i32,
    output_dir: &PathBuf,
) -> Result<(), String> {
    info!("Generating selection overlay PNG with tiles");
    
    let (src_w, src_h) = ds.raster_size();
    
    // Scale down if too large for preview
    let (display_width, display_height) = if src_w > 2000 || src_h > 2000 {
        let scale = ((src_w as f64).max(src_h as f64) / 2000.0).ceil() as usize;
        ((src_w as usize) / scale, (src_h as usize) / scale)
    } else {
        (src_w as usize, src_h as usize)
    };
    
    info!("Creating overlay with full dataset view: {} x {} pixels", display_width, display_height);
    
    // Read the first band
    let band1 = ds.rasterband(1).map_err(|e| e.to_string())?;
    let buffer: Buffer<f32> = band1
        .read_as(
            (0, 0),
            (src_w as usize, src_h as usize),
            (display_width, display_height),
            None,
        )
        .map_err(|e| e.to_string())?;
    
    let data = buffer.data();
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = if (max_val - min_val).abs() > 0.001 {
        max_val - min_val
    } else {
        1.0
    };
    
    info!("Data range: {:.2} to {:.2}", min_val, max_val);
    
    // Create RGBA image from the grayscale data (for drawing operations)
    let mut img: image::RgbaImage = ImageBuffer::new(display_width as u32, display_height as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let idx = (y as usize * display_width + x as usize) as usize;
        if idx < data.len() {
            let normalized = ((data[idx] - min_val) / range * 255.0).clamp(0.0, 255.0) as u8;
            *pixel = image::Rgba([normalized, normalized, normalized, 255]);
        }
    }
    
    // Calculate the scaled selection rectangle (red)
    let scale_factor = display_width as f64 / src_w as f64;
    let rect_x = (x_off as f64 * scale_factor).round() as i32;
    let rect_y = (y_off as f64 * scale_factor).round() as i32;
    let rect_width = (width as f64 * scale_factor).round() as u32;
    let rect_height = (height as f64 * scale_factor).round() as u32;
    
    info!("Drawing selection rectangle at ({}, {}) with size {} x {}", 
          rect_x, rect_y, rect_width, rect_height);
    
    // Draw selection rectangle in red
    let red = image::Rgba([255u8, 0u8, 0u8, 255u8]);
    let rect_x_clamp = rect_x.max(0) as u32;
    let rect_y_clamp = rect_y.max(0) as u32;
    let rect_x_max = ((rect_x as u32 + rect_width).min(display_width as u32)) as i32;
    let rect_y_max = ((rect_y as u32 + rect_height).min(display_height as u32)) as i32;
    
    // Draw top and bottom lines
    for x in rect_x_clamp..(rect_x_max.max(0) as u32).min(display_width as u32) {
        if rect_y >= 0 && rect_y < display_height as i32 {
            *img.get_pixel_mut(x, rect_y as u32) = red;
        }
        if rect_y_max > 0 && rect_y_max < display_height as i32 {
            *img.get_pixel_mut(x, rect_y_max as u32) = red;
        }
    }
    
    // Draw left and right lines
    for y in rect_y_clamp..(rect_y_max.max(0) as u32).min(display_height as u32) {
        if rect_x >= 0 && rect_x < display_width as i32 {
            *img.get_pixel_mut(rect_x as u32, y) = red;
        }
        if rect_x_max > 0 && rect_x_max < display_width as i32 {
            *img.get_pixel_mut(rect_x_max as u32, y) = red;
        }
    }
    
    // Draw tile grid in white
    let white = image::Rgba([255u8, 255u8, 255u8, 255u8]);
    let tile_px_scaled = (tile_px as f64 * scale_factor) as i32;
    
    // Draw vertical grid lines
    for col in 1..tiles_x {
        let x_pos = rect_x + col as i32 * tile_px_scaled;
        if x_pos >= 0 && x_pos < display_width as i32 {
            for y in rect_y_clamp..(rect_y_max.max(0) as u32).min(display_height as u32) {
                if x_pos >= 0 && x_pos < display_width as i32 {
                    *img.get_pixel_mut(x_pos as u32, y) = white;
                }
            }
        }
    }
    
    // Draw horizontal grid lines
    for row in 1..tiles_y {
        let y_pos = rect_y + row as i32 * tile_px_scaled;
        if y_pos >= 0 && y_pos < display_height as i32 {
            for x in rect_x_clamp..(rect_x_max.max(0) as u32).min(display_width as u32) {
                if y_pos >= 0 && y_pos < display_height as i32 {
                    *img.get_pixel_mut(x, y_pos as u32) = white;
                }
            }
        }
    }
    
    // Save PNG
    let overlay_path = output_dir.join("selection_overlay.png");
    img.save(&overlay_path).map_err(|e| e.to_string())?;
    info!("Selection overlay PNG saved to {}", overlay_path.display());
    
    Ok(())
}