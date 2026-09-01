# Import library to parse command-line arguments
import argparse
# Import Path to handle cross-platform file paths safely
from pathlib import Path
# Import warnings to silence excessive console output from Astropy
import warnings
# Import ThreadPoolExecutor to enable parallel downloads
from concurrent.futures import ThreadPoolExecutor

# Import numpy for numerical operations and array handling
import numpy as np
# Import matplotlib main library to change global parameters
import matplotlib as mpl
# Import pyplot for plotting graphs and images
import matplotlib.pyplot as plt
# Import Circle and Rectangle to draw the Field of View and Slit
from matplotlib.patches import Circle, Rectangle

# Import Astropy units for handling physical quantities like degrees
import astropy.units as u
# Import SkyCoord to handle astronomical coordinates
from astropy.coordinates import SkyCoord
# Import WCS to handle World Coordinate System transformations
from astropy.wcs import WCS
# Import Astropy warnings to properly filter them
from astropy.utils.exceptions import AstropyWarning
# Import visualization tools to adjust image contrast
from astropy.visualization import ImageNormalize, ZScaleInterval

# Import reproject to align image data to a new WCS
from reproject import reproject_interp

# Import custom functions from the local utils.py module
from utils import (
    query_stars_gaia, query_stars_ps1, query_stars_ls, get_stars_2mass,
    parse_coords, get_image_2mass, get_image_fallbacks, setup_logger, upload_to_drive
)

# Silence all Astropy and general user warnings for a clean terminal output
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)

# Set the global font size default for matplotlib plots to 15
mpl.rcParams["font.size"] = 15

# Define function to retrieve stars from optical catalogs with automatic failover
def get_stars_optical(ra, dec, radius=3):
    # 1. Try Gaia first
    try:
        stars = query_stars_gaia(ra, dec, radius=radius)
        if not stars.empty:
            print('Star catalog from Gaia')
            return stars
    except Exception as e:
        print(f"⚠️ Warning: Gaia query failed. ({e}). Trying PS1...")

    # 2. Try Pan-STARRS 1 (valid for declinations greater than -30)
    if dec > -30:
        try:
            stars = query_stars_ps1(ra, dec, radius=radius)
            if not stars.empty:
                print('Star catalog from PS1')
                return stars
        except Exception as e:
            print(f"⚠️ Warning: PS1 query failed. ({e}). Trying LS...")

    # 3. Try Legacy Survey (valid for declinations less than 30)
    if dec < 30:
        try:
            stars = query_stars_ls(ra, dec, radius=radius)
            if not stars.empty:
                print('Star catalog from LS')
                return stars
        except Exception as e:
            print(f"⚠️ Warning: LS query failed. ({e}).")

    # 4. If all catalogs fail
    print('❌ Could not retrieve reference stars from any optical catalog.')
    return ''

# Define function to select the appropriate star catalog based on wavelength
def get_stars(ra, dec, radius=3, wv='optical'):
    if wv == 'optical':
        stars = get_stars_optical(ra, dec, radius=radius)
    elif wv == 'ir':
        stars = get_stars_2mass(ra, dec, radius=radius)
        if not stars.empty:
            print('Star catalog from 2MASS')
        else:
            print('Failed to retrieve stars from 2MASS, trying optical surveys')
            stars = get_stars_optical(ra, dec, radius=radius)
    else:
        print(f'{wv} wavelength not supported')
        return ''
        
    if isinstance(stars, str) or stars.empty:
        return ''
        
    target = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    coords = SkyCoord(stars["ra"].values * u.deg, stars["dec"].values * u.deg, frame="icrs")
    
    dra, ddec = target.spherical_offsets_to(coords)
    stars["offset_EW_arcsec"] = dra.to(u.arcsec).value
    stars["offset_NS_arcsec"] = ddec.to(u.arcsec).value

    return stars

# Modular Plotting Function: Draw a North-East compass rose
def add_compass_rose(ax, visible_size, cx, cy, wcs, is_rotated=False):
    length = visible_size * 0.08
    gold = "#E69F00"
    margin = visible_size * 0.10
    
    if not is_rotated:
        x0 = (cx - visible_size / 2) + margin
        y0 = (cy + visible_size / 2) - margin
    else:
        x0 = (cx + visible_size / 2) - margin
        y0 = (cy - visible_size / 2) + margin
        
    pix_origin = wcs.wcs.crpix
    world_origin = SkyCoord(wcs.wcs.crval[0] * u.deg, wcs.wcs.crval[1] * u.deg, frame="icrs")
    
    p_n = world_origin.directional_offset_by(0 * u.deg, 1 * u.arcmin)
    pix_n = wcs.world_to_pixel(p_n)
    dx_n, dy_n = pix_n[0] - pix_origin[0], pix_n[1] - pix_origin[1]
    
    p_e = world_origin.directional_offset_by(90 * u.deg, 1 * u.arcmin)
    pix_e = wcs.world_to_pixel(p_e)
    dx_e, dy_e = pix_e[0] - pix_origin[0], pix_e[1] - pix_origin[1]

    def norm_v(dx, dy):
        mag = np.sqrt(dx**2 + dy**2)
        return (dx / mag) * length, (dy / mag) * length if mag != 0 else (0, 0)

    dnx, dny = norm_v(dx_n, dy_n)
    dex, dey = norm_v(dx_e, dy_e)

    ax.arrow(x0, y0, dnx, dny, color=gold, width=visible_size*0.002, head_width=visible_size*0.015, zorder=20)
    ax.text(x0 + dnx*1.6, y0 + dny*1.6, "N", color=gold, ha="center", va="center", fontweight="bold", zorder=20)
    
    ax.arrow(x0, y0, dex, dey, color=gold, width=visible_size*0.002, head_width=visible_size*0.015, zorder=20)
    ax.text(x0 + dex*1.6, y0 + dey*1.6, "E", color=gold, ha="center", va="center", fontweight="bold", zorder=20)

# Modular Plotting Function: Draw symmetrical crosshairs
def draw_crosshair(ax, x, y, gap, arm, color, label=None, label_offset=0):
    ax.plot([x + gap, x + arm], [y, y], color=color, lw=3 if not label else 2)
    ax.plot([x - arm, x - gap], [y, y], color=color, lw=3 if not label else 2)
    ax.plot([x, x], [y + gap, y + arm], color=color, lw=3 if not label else 2)
    ax.plot([x, x], [y - arm, y - gap], color=color, lw=3 if not label else 2)
    if label:
        ax.text(x + arm + label_offset, y + arm + label_offset, label, color=color, fontsize=12, fontweight='bold')

# Modular Plotting Function: Draw scale bar (Updated to fix overlapping text)
def draw_scale_bar(ax, cx, cy, target_npix, pixscale, is_rotated=False):
    bar_px = 60 / pixscale
    bx0 = (cx - target_npix/2) + (target_npix * 0.05)
    by0 = (cy - target_npix/2) + (target_npix * 0.05)
    ax.plot([bx0, bx0 + bar_px], [by0, by0], color='blue', lw=3)
    
    # Adjust text position depending on rotation to avoid overlapping the line
    if is_rotated:
        ax.text(bx0 + bar_px/2, by0 + (target_npix * 0.03), "1'", color='blue', ha='center', va='top', fontweight='bold')
    else:
        ax.text(bx0 + bar_px/2, by0 + (target_npix * 0.03), "1'", color='blue', ha='center', va='bottom', fontweight='bold')

# Define main function to build the figure from FITS files
def fits2image_projected(hdu_opt, hdu_ir, stars_opt, stars_ir, pa_deg=0, imsize=6, radius=3, contrast=0.045):
    fig = plt.figure(figsize=(22, 16), constrained_layout=False)
    spec = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[4, 4, 2.8], height_ratios=[1, 1],
                            left=0.05, right=0.95, wspace=0.15, hspace=0.2)

    ax_text = fig.add_subplot(spec[:, 2])
    ax_text.axis("off")

    base_hdu = hdu_opt if hdu_opt else hdu_ir
    s_name = base_hdu[0].header['s_name']
    base_ra, base_dec = base_hdu[0].header['ra'], base_hdu[0].header['dec']
    
    # Emphasis on Target Name and Coordinates
    ax_text.text(0, 0.97, f"TARGET: {s_name}", color="#8B0000", fontsize=22, fontweight="bold")
    ax_text.text(0, 0.92, f"RA: {base_ra:.5f}\nDEC: {base_dec:.5f}", color="#000080", fontsize=16, fontweight="bold")

    # Sub-function to plot individual ROW (Optical top, IR bottom)
    def plot_row(hdu, row_idx, catalog_name, filter_name, prefix_dir, prefix_rot, text_y_start, color_dir, color_rot, num_dir, num_rot, stars_df):
        pixscale = hdu[0].header['pixscale']
        ra, dec = hdu[0].header['ra'], hdu[0].header['dec']
        npixels = hdu[0].header['numpix']
        
        wcs = WCS(naxis=2)
        wcs.wcs.crpix, wcs.wcs.crval = [npixels / 2, npixels / 2], [ra, dec]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        pa_rad = np.deg2rad(pa_deg)
        rot = np.array([[np.cos(pa_rad), np.sin(pa_rad)], [-np.sin(pa_rad), np.cos(pa_rad)]])
        wcs.wcs.cd = rot @ np.array([[-pixscale / 3600, 0], [0, pixscale / 3600]])

        im, _ = reproject_interp((hdu[0].data, WCS(hdu[0].header)), wcs, shape_out=(npixels, npixels))
        
        # Protection against completely empty images (outside the catalog's coverage area)
        if np.all(np.isnan(im)):
            print(f"⚠️ Warning: The image for {catalog_name} is empty (possibly outside coverage area).")
            im = np.zeros((int(npixels), int(npixels))) # Creates a black background
            norm = None
        else:
            im[np.isnan(im)] = np.nanmedian(im)
            try:
                norm = ImageNormalize(im, interval=ZScaleInterval(contrast=contrast))
            except IndexError:
                # Fallback in case the image has data but is completely flat
                norm = None

        ax_dir = fig.add_subplot(spec[row_idx, 0], projection=wcs)
        ax_rot = fig.add_subplot(spec[row_idx, 1], projection=wcs)

        target_npix = (imsize * 60) / pixscale
        cx, cy = npixels / 2, npixels / 2

        for plot_ax in [ax_dir, ax_rot]:
            plot_ax.imshow(im, origin="lower", norm=norm, cmap="gray_r")
            plot_ax.set_autoscale_on(False)
            
            plot_ax.coords[0].set_major_formatter('hh:mm:ss')
            plot_ax.coords[1].set_major_formatter('dd:mm:ss')
            
            if plot_ax == ax_rot:
                plot_ax.invert_xaxis()
                plot_ax.invert_yaxis()
                plot_ax.set_xlim(cx + target_npix/2, cx - target_npix/2)
                plot_ax.set_ylim(cy + target_npix/2, cy - target_npix/2)
                plot_ax.set_title(f"{num_rot} | {s_name} | {catalog_name} ({filter_name}) | PA: {(pa_deg + 180) % 360}°", color=color_rot, fontweight="bold", loc='right', fontsize=11)
            else:
                plot_ax.set_xlim(cx - target_npix/2, cx + target_npix/2)
                plot_ax.set_ylim(cy - target_npix/2, cy + target_npix/2)
                plot_ax.set_title(f"{num_dir} | {s_name} | {catalog_name} ({filter_name}) | PA: {pa_deg}°", color=color_dir, fontweight="bold", loc='right', fontsize=11)

            plot_ax.grid(color="white", ls="dotted", alpha=0.5)
            plot_ax.set_xlabel(r"$\alpha$ (J2000)", fontsize="large")
            plot_ax.set_ylabel(r"$\delta$ (J2000)", fontsize="large")

            add_compass_rose(plot_ax, target_npix, cx, cy, wcs, is_rotated=(plot_ax == ax_rot))

            tx, ty = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"))
            draw_crosshair(plot_ax, tx, ty, gap=4.0/pixscale, arm=12.0/pixscale, color="#D55E00")
            plot_ax.add_patch(Circle((tx, ty), radius=1.0/pixscale, edgecolor='#D55E00', facecolor='none', lw=1.5, ls='--'))

            # Spectrograph Slit Overlay (1.0" width x 4.0' height)
            slit_w_px = 1.0 / pixscale
            slit_h_px = 240.0 / pixscale # 4 arcminutes in arcseconds
            # Since the array 'im' was generated with the Y-axis aligned to the PA, the slit is always vertical.
            slit_rect = Rectangle((tx - slit_w_px/2, ty - slit_h_px/2), slit_w_px, slit_h_px, 
                                  facecolor='green', edgecolor='lime', alpha=0.15, lw=1.5, zorder=5)
            plot_ax.add_patch(slit_rect)

        # Scale bars on both direct and rotated plots (passing is_rotated flag)
        draw_scale_bar(ax_dir, cx, cy, target_npix, pixscale, is_rotated=False)
        draw_scale_bar(ax_rot, cx, cy, target_npix, pixscale, is_rotated=True)

        stars = stars_df
        if not isinstance(stars, str) and not stars.empty:
            # Intense colors for reference stars
            top3, colors = stars.iloc[:3], ["#FFD700", "#00BFFF", "#FF00FF"]
            
            ax_text.text(0, text_y_start, f"{catalog_name} Ref Stars:", fontweight="bold", fontsize=12)
            
            s_gap, s_arm, t_offset = 2.5/pixscale, 7.0/pixscale, 3.0/pixscale

            # --- DIRECT OFFSETS INFO (Table Format) ---
            ax_text.text(0, text_y_start - 0.05, f"{num_dir} - Direct Offsets (PA: {pa_deg}°):", color=color_dir, fontweight="bold", fontsize=11)
            for i, (_, row) in enumerate(top3.iterrows()):
                sx, sy = wcs.world_to_pixel(SkyCoord(row.ra * u.deg, row.dec * u.deg, frame="icrs"))
                draw_crosshair(ax_dir, sx, sy, gap=s_gap, arm=s_arm, color=colors[i], label=f"{prefix_dir}{i+1}", label_offset=t_offset)
                
                ew_dir = 'W' if row.offset_EW_arcsec >= 0 else 'E'
                ns_dir = 'S' if row.offset_NS_arcsec >= 0 else 'N'
                
                # Increased the vertical gap to fit the new coordinate line below the offsets
                y_pos_main = text_y_start - 0.10 - (i * 0.09)
                y_pos_coords = y_pos_main - 0.04
                
                # Column 1: Star Label
                ax_text.text(0.00, y_pos_main, rf"$\bf{{{prefix_dir}{i+1}}}$", color=colors[i], fontsize=13)
                # Column 2: Magnitude
                ax_text.text(0.10, y_pos_main, f"{row.mag:.1f}m", color=colors[i], fontsize=13)
                # Column 3: EW Offset
                ax_text.text(0.35, y_pos_main, rf"$\bf{{{abs(row.offset_EW_arcsec):.1f}''\ {ew_dir}}}$", color=colors[i], fontsize=14)
                # Column 4: NS Offset
                ax_text.text(0.70, y_pos_main, rf"$\bf{{{abs(row.offset_NS_arcsec):.1f}''\ {ns_dir}}}$", color=colors[i], fontsize=14)
                
                # New Column/Row: Star Coordinates
                ax_text.text(0.10, y_pos_coords, f"RA: {row.ra:.5f}°, Dec: {row.dec:.5f}°", color=colors[i], fontsize=11)

            # Adjusted return height since the rotated block was deleted
            return text_y_start - 0.35
            
        return text_y_start

    # Adjust starting Y position to accommodate larger Target header
    current_y = 0.83
    
    if hdu_opt:
        cat_name = hdu_opt[0].header.get('w_mark', 'Optical')
        filter_name = "Red" if cat_name == "DSS" else "r-band"
        current_y = plot_row(hdu_opt, 0, cat_name, filter_name, "a", "b", current_y, color_dir="#0033CC", color_rot="#CC0000", num_dir="I", num_rot="II", stars_df=stars_opt)
        
    if hdu_ir:
        plot_row(hdu_ir, 1, "2MASS", "J-band", "c", "d", current_y, color_dir="#008000", color_rot="#800080", num_dir="III", num_rot="IV", stars_df=stars_ir)

    return fig

# Main execution pipeline logic (callable directly from run_batch)
def run_pipeline(s_name, ra_str, dec_str, pa_deg=0.0, imsize=4.0, radius=1.0, contrast=0.045, output_folder='./finder_charts/', drive_folder=None):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(name="finder chart", logfile="./finder.log")
    logger.info(f"Pipeline started for {s_name}")
    
    ra, dec = parse_coords(ra_str, dec_str)
    print(f"Source: {s_name} | RA, Dec: {ra:.6f}, {dec:.6f} | PA: {pa_deg} | FOV: {imsize}'")
    
    download_imsize = imsize * 1.5
    hdu_opt, hdu_ir = None, None
    stars_opt, stars_ir = '', ''

    # --- MULTITHREADING IMPLEMENTATION (Queries & Images Concurrent) ---
    print("Downloading images and querying star catalogs in parallel...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Launch all 4 tasks concurrently
        future_opt_img = executor.submit(get_image_fallbacks, ra, dec, s_name, download_imsize)
        # future_ir_img = executor.submit(get_image_2mass, ra, dec, s_name, download_imsize)
        future_opt_stars = executor.submit(get_stars, ra, dec, radius, 'optical')
        # future_ir_stars = executor.submit(get_stars, ra, dec, radius, 'ir')
        
        # Retrieve Optical Results
        try:
            hdu_opt = future_opt_img.result()
            if hdu_opt:
                hdu_opt[0].header['wv'] = 'optical'
                print(f"Optical image fetched successfully from {hdu_opt[0].header.get('w_mark', 'Catalog')}")
            stars_opt = future_opt_stars.result()
        except Exception as e:
            print(f"Warning: Could not fetch optical image or stars. {e}")
            
        # Retrieve IR Results
        # try:
        #     hdu_ir = future_ir_img.result()
        #     if hdu_ir:
        #         hdu_ir[0].header['wv'] = 'ir'
        #         print("Infrared image fetched successfully from 2MASS")
        #     stars_ir = future_ir_stars.result()
        # except Exception as e:
        #     print(f"Warning: Could not fetch 2MASS image or stars. {e}")
    # ------------------------------------------------------------------

    if not hdu_opt and not hdu_ir:
        raise ValueError("Could not fetch ANY images for this target.")
            
    print("Generating Finder Chart PDF...")
    # fig = fits2image_projected(hdu_opt, hdu_ir, stars_opt, stars_ir, pa_deg=pa_deg, imsize=imsize, radius=radius, contrast=contrast)
    fig = fits2image_projected(hdu_opt, hdu_opt, stars_opt, stars_opt, pa_deg=pa_deg, imsize=imsize, radius=radius, contrast=contrast)
    
    output_file = Path(output_folder) / f'{s_name}.pdf'
    fig.savefig(output_file, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Success! PDF saved to: {output_file}")
    
    # --- GOOGLE DRIVE UPLOAD ---
    if drive_folder:
        upload_to_drive(output_file, drive_folder)

# CLI Parser function
def parse_args():
    parser = argparse.ArgumentParser(description="Make a finder for an object")
    parser.add_argument("--ra", dest="ra_", required=True, help="Right Ascension (degree or sexagesimal)")
    parser.add_argument("--dec", dest="dec_", required=True, help="Declination (degree or sexagesimal)")
    parser.add_argument("--s-name", default="Target", help="Source name (string)")
    parser.add_argument("--pa-deg", type=float, default=0.0, help="Position angle in degrees (East of North)")
    parser.add_argument("--imsize", type=float, default=4.0, help="Image size in arcminutes")
    parser.add_argument("--radius", type=float, default=1.0, help="Radius in arcminutes")
    parser.add_argument("--output-folder", type=str, default='./finder_charts/', help="output folder")
    parser.add_argument("--contrast", type=float, default=0.045, help="Contrast for ZScaleInterval")
    parser.add_argument("--drive-folder", type=str, default=None, help="Google Drive Folder ID to upload the PDF")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        s_name=args.s_name, ra_str=args.ra_, dec_str=args.dec_, pa_deg=args.pa_deg, 
        imsize=args.imsize, radius=args.radius, contrast=args.contrast, 
        output_folder=args.output_folder, drive_folder=args.drive_folder
    )
