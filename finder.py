import argparse
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib as mpl
# Force Matplotlib to use the 'Agg' backend (Headless mode) to prevent server crashes
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import ImageNormalize, ZScaleInterval
from reproject import reproject_interp

# Import utilities for data fetching and Drive uploads
from utils import (query_stars_gaia, query_stars_ps1, query_stars_ls, get_stars_2mass, parse_coords, get_image_2mass, get_image_fallbacks, setup_logger, upload_to_drive)

# Suppress annoying warnings from Astropy regarding WCS headers
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)
# Set default font size for the plots
mpl.rcParams["font.size"] = 15

# Initialize logger
logger = setup_logger(name="finder_engine")

def get_stars_optical(ra, dec, radius=3.0):
    # Loop through catalogs in order of preference: Gaia -> PS1 -> LS
    for func, name in [(query_stars_gaia, "Gaia"), (query_stars_ps1, "Pan-STARRS"), (query_stars_ls, "Legacy Survey")]:
        try:
            # Attempt to fetch stars
            stars = func(ra, dec, radius=radius)
            # If successful and not empty, return them immediately
            if stars is not None and not stars.empty: 
                return stars
        except Exception as e: 
            # Log failure and proceed to the next fallback catalog
            logger.warning(f"{name} query failed: {e}")
    # Return empty string if all fail
    return ''

def get_stars(ra, dec, radius=3.0, wv='optical'):
    # Force a large 7.0 arcminute search to capture distant stars in one network request
    search_radius = 7.0 
    logger.info(f"Querying {wv.upper()} reference stars within {search_radius}'...")
    
    # Query optical or IR based on requested wavelength
    stars = get_stars_optical(ra, dec, search_radius) if wv == 'optical' else get_stars_2mass(ra, dec, search_radius)
    
    # If IR (2MASS) fails, fallback to optical stars for the IR plot
    if wv == 'ir' and (isinstance(stars, str) or stars.empty):
        stars = get_stars_optical(ra, dec, search_radius)

    # If we successfully found stars
    if not isinstance(stars, str) and not stars.empty:
        # Create SkyCoord objects for the target and the found stars
        target = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        coords = SkyCoord(stars["ra"].values * u.deg, stars["dec"].values * u.deg, frame="icrs")
        
        # Calculate spherical offsets (North/South, East/West)
        dra, ddec = target.spherical_offsets_to(coords)
        stars["offset_EW_arcsec"], stars["offset_NS_arcsec"] = dra.to(u.arcsec).value, ddec.to(u.arcsec).value
        
        # Calculate total absolute distance (Hypotenuse)
        stars["total_dist_arcsec"] = np.sqrt(stars["offset_EW_arcsec"]**2 + stars["offset_NS_arcsec"]**2)
        
        # Exclude stars that are closer than 2.0 arcseconds to the target (blending prevention)
        stars = stars[stars["total_dist_arcsec"] >= 2.0]
        
        # Sort by distance (closest first) and return
        return stars.sort_values(by="total_dist_arcsec").reset_index(drop=True)
    return ''

def add_compass_rose(ax, visible_size, cx, cy, wcs, is_rotated=False):
    # Calculate lengths and margins for the compass arrows
    length, margin = visible_size * 0.08, visible_size * 0.10
    # Determine placement based on whether the image is inverted
    x0 = (cx + visible_size / 2) - margin if is_rotated else (cx - visible_size / 2) + margin
    y0 = (cy - visible_size / 2) + margin if is_rotated else (cy + visible_size / 2) - margin
    
    # Calculate pixel offsets corresponding to True North and East using WCS
    world_origin = SkyCoord(wcs.wcs.crval[0] * u.deg, wcs.wcs.crval[1] * u.deg, frame="icrs")
    def get_vec(ang):
        p = wcs.world_to_pixel(world_origin.directional_offset_by(ang, 1 * u.arcmin))
        dx, dy = p[0] - wcs.wcs.crpix[0], p[1] - wcs.wcs.crpix[1]
        mag = np.sqrt(dx**2 + dy**2)
        return (dx / mag) * length, (dy / mag) * length if mag != 0 else (0, 0)

    dnx, dny = get_vec(0 * u.deg)
    dex, dey = get_vec(90 * u.deg)

    # Draw the arrows and text labels
    for dx, dy, label in [(dnx, dny, "N"), (dex, dey, "E")]:
        ax.arrow(x0, y0, dx, dy, color="#E69F00", width=visible_size*0.002, head_width=visible_size*0.015, zorder=20)
        ax.text(x0 + dx*1.6, y0 + dy*1.6, label, color="#E69F00", ha="center", va="center", fontweight="bold", zorder=20)

def draw_crosshair(ax, x, y, gap, arm, color, label=None, offset=0):
    # Draw four lines forming a broken crosshair around the coordinates
    for dx1, dx2, dy1, dy2 in [(gap, arm, 0, 0), (-arm, -gap, 0, 0), (0, 0, gap, arm), (0, 0, -arm, -gap)]:
        ax.plot([x + dx1, x + dx2], [y + dy1, y + dy2], color=color, lw=3 if not label else 2)
    # Add an optional text label (e.g., "a1", "b2")
    if label: 
        ax.text(x + arm + offset, y + arm + offset, label, color=color, fontsize=12, fontweight='bold')

def draw_scale_bar(ax, cx, cy, target_npix, pixscale, is_rotated=False):
    # Calculate pixels for 1 arcminute (60 arcseconds)
    bar_px, bx0, by0 = 60 / pixscale, (cx - target_npix/2) + (target_npix * 0.05), (cy - target_npix/2) + (target_npix * 0.05)
    # Draw the blue line
    ax.plot([bx0, bx0 + bar_px], [by0, by0], color='blue', lw=3)
    # Add the "1'" text
    ax.text(bx0 + bar_px/2, by0 + (target_npix * 0.03), "1'", color='blue', ha='center', va='top' if is_rotated else 'bottom', fontweight='bold')

def fits2image_projected(hdu_opt, hdu_ir, stars_opt, stars_ir, pa_deg=0, imsize=3.0, slit_width=1.0, slit_height=234.0, is_parallactic=False):
    # Create the main figure canvas
    fig = plt.figure(figsize=(22, 16))
    # Define a grid layout: 2 rows, 3 columns (Left Image, Rotated Image, Text Box)
    spec = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[4, 4, 2.8], left=0.05, right=0.95, wspace=0.15, hspace=0.2)
    
    # Set up the text box on the far right
    ax_text = fig.add_subplot(spec[:, 2]); ax_text.axis("off"); ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)

    # Extract header info from whichever FITS file successfully downloaded
    base_hdu = hdu_opt if hdu_opt else hdu_ir
    s_name, base_ra, base_dec = base_hdu[0].header['s_name'], base_hdu[0].header['ra'], base_hdu[0].header['dec']
    
    # Write the main Title and Coordinates
    ax_text.text(0, 0.97, f"TARGET: {s_name}", color="#8B0000", fontsize=22, fontweight="bold")
    ax_text.text(0, 0.92, f"RA: {base_ra:.5f}\nDEC: {base_dec:.5f}", color="#000080", fontsize=16, fontweight="bold")
    
    # Print a warning if the observer requested parallactic angle
    if is_parallactic: 
        ax_text.text(0, 0.87, "⚠️ ROTATE TO PARALLACTIC ⚠️", color="red", fontsize=14, fontweight="bold")

    # Inner helper function to plot a single row (e.g., Optical or IR row)
    def plot_row(hdu, row_idx, cat_name, filt, p_dir, p_rot, y_start, c_dir, c_rot, n_dir, n_rot, stars_df):
        pix, ra, dec, npix = hdu[0].header['pixscale'], hdu[0].header['ra'], hdu[0].header['dec'], hdu[0].header['numpix']
        
        # Construct a synthetic WCS to handle the requested rotation (PA)
        wcs = WCS(naxis=2)
        wcs.wcs.crpix, wcs.wcs.crval, wcs.wcs.ctype = [npix / 2, npix / 2], [ra, dec], ["RA---TAN", "DEC--TAN"]
        pa_rad = np.deg2rad(pa_deg)
        # Apply the rotation matrix to the WCS CD matrix
        wcs.wcs.cd = np.array([[np.cos(pa_rad), np.sin(pa_rad)], [-np.sin(pa_rad), np.cos(pa_rad)]]) @ np.array([[-pix / 3600, 0], [0, pix / 3600]])

        # Reproject the original image onto our new rotated WCS grid
        im, _ = reproject_interp((hdu[0].data, WCS(hdu[0].header)), wcs, shape_out=(npix, npix))
        # Normalize the image contrast using ZScale
        norm = None if np.all(np.isnan(im)) else ImageNormalize(np.nan_to_num(im, nan=np.nanmedian(im)), interval=ZScaleInterval(contrast=0.045))

        # Create the two plot axes for this row
        ax_dir, ax_rot = fig.add_subplot(spec[row_idx, 0], projection=wcs), fig.add_subplot(spec[row_idx, 1], projection=wcs)
        cx, target_npix = npix / 2, (imsize * 60) / pix

        # Loop through both plots (Direct View vs Rotated View)
        for ax, is_rot, num, col, pa_val in [(ax_dir, False, n_dir, c_dir, pa_deg), (ax_rot, True, n_rot, c_rot, (pa_deg+180)%360)]:
            ax.imshow(im, origin="lower", norm=norm, cmap="gray_r")
            
            # Invert X and Y for the Rotated view to simulate standard telescope flipping
            if is_rot: 
                ax.invert_xaxis(); ax.invert_yaxis()
            
            # Crop the plot limits to exactly match the requested FOV
            lim_sign = -1 if is_rot else 1
            ax.set_xlim(cx - lim_sign*target_npix/2, cx + lim_sign*target_npix/2)
            ax.set_ylim(cx - lim_sign*target_npix/2, cx + lim_sign*target_npix/2)
            
            # Set titles and grids
            ax.set_title(f"{num} | {s_name} | {cat_name} ({filt}) | PA: {pa_val}°", color=col, fontweight="bold", loc='right')
            ax.grid(color="white", ls="dotted", alpha=0.5)
            
            # Draw overlays
            add_compass_rose(ax, target_npix, cx, cx, wcs, is_rotated=is_rot)
            draw_scale_bar(ax, cx, cx, target_npix, pix, is_rotated=is_rot)

            # Mark the science target exactly in the center
            tx, ty = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"))
            draw_crosshair(ax, tx, ty, gap=4.0/pix, arm=12.0/pix, color="#D55E00")
            ax.add_patch(Circle((tx, ty), radius=1.0/pix, edgecolor='#D55E00', facecolor='none', lw=1.5, ls='--'))
            
            # Draw the green translucent Slit based on instrument specs
            ax.add_patch(Rectangle((tx - (slit_width/pix)/2, ty - (slit_height/pix)/2), slit_width/pix, slit_height/pix, facecolor='green', edgecolor='lime', alpha=0.15, zorder=5))

        # If reference stars exist, draw them and print the text table
        if not isinstance(stars_df, str) and not stars_df.empty:
            colors = ["#FFD700", "#00BFFF", "#FF00FF"]
            ax_text.text(0, y_start, f"{cat_name} Ref Stars:", fontweight="bold", fontsize=12)
            
            # Loop for both views (Direct and Rotated offsets)
            for k, (ax, pfx, col, y_off, is_rot) in enumerate([(ax_dir, p_dir, c_dir, 0.05, False), (ax_rot, p_rot, c_rot, 0.26, True)]):
                ax_text.text(0, y_start - y_off, f"Offsets (PA: {(pa_deg+180)%360 if is_rot else pa_deg}°):", color=col, fontweight="bold", fontsize=11)
                
                # Loop through the top 3 closest stars
                for i, (_, row) in enumerate(stars_df.head(3).iterrows()):
                    # Mark star on image
                    sx, sy = wcs.world_to_pixel(SkyCoord(row.ra * u.deg, row.dec * u.deg, frame="icrs"))
                    draw_crosshair(ax, sx, sy, gap=2.5/pix, arm=7.0/pix, color=colors[i], label=f"{pfx}{i+1}", offset=3.0/pix)
                    
                    # Flip direction letters if the image is inverted
                    ew = ('E' if row.offset_EW_arcsec >= 0 else 'W') if is_rot else ('W' if row.offset_EW_arcsec >= 0 else 'E')
                    ns = ('N' if row.offset_NS_arcsec >= 0 else 'S') if is_rot else ('S' if row.offset_NS_arcsec >= 0 else 'N')
                    
                    # Print table row
                    y_p = y_start - y_off - 0.05 - (i * 0.045)
                    ax_text.text(0.00, y_p, rf"$\bf{{{pfx}{i+1}}}$", color=colors[i], fontsize=13)
                    ax_text.text(0.10, y_p, f"{row.mag:.1f}m", color=colors[i], fontsize=13)
                    ax_text.text(0.35, y_p, rf"$\bf{{{abs(row.offset_EW_arcsec):.1f}''\ {ew}}}$", color=colors[i], fontsize=14)
                    ax_text.text(0.70, y_p, rf"$\bf{{{abs(row.offset_NS_arcsec):.1f}''\ {ns}}}$", color=colors[i], fontsize=14)
            return y_start - 0.48
    
    # Call the row plotting function for Optical and IR data (if successfully downloaded)
    if hdu_opt: 
        plot_row(hdu_opt, 0, hdu_opt[0].header.get('w_mark', 'Optical'), "Red" if hdu_opt[0].header.get('w_mark') == "DSS" else "r-band", "a", "b", 0.83, "#0033CC", "#CC0000", "I", "II", stars_opt)
    if hdu_ir: 
        plot_row(hdu_ir, 1, "2MASS", "J-band", "c", "d", 0.38, "#008000", "#800080", "III", "IV", stars_ir)
    return fig

def run_pipeline(s_name, ra_str, dec_str, instrument="GOODMAN", pa_deg=0.0, imsize=3.0, radius=1.0, contrast=0.045, slit_width=1.0, output_folder='./finder_charts/', drive_folder=None, is_parallactic=False):
    # Parse coordinates into decimal degrees
    ra, dec = parse_coords(ra_str, dec_str)
    
    # AEON Instrument Dictionary configuring physical Slit Height and minimum valid FOV
    INSTRUMENT_SPECS = {
        'GOODMAN': {'slit_h': 234.0, 'min_fov': 2.0},
        'GMOS':    {'slit_h': 330.0, 'min_fov': 5.5}, 
        'DEFAULT': {'slit_h': 234.0, 'min_fov': 2.0}
    }
    # Retrieve specs, defaulting to GOODMAN if unknown
    specs = INSTRUMENT_SPECS.get(str(instrument).upper(), INSTRUMENT_SPECS['DEFAULT'])

    # Query Optical and IR stars concurrently to save network time
    with ThreadPoolExecutor(max_workers=2) as executor:
        stars_opt = executor.submit(get_stars, ra, dec, 7.0, 'optical').result()
        stars_ir = executor.submit(get_stars, ra, dec, 7.0, 'ir').result()

    # Calculate the maximum distance of the selected top 3 stars to determine the required FOV
    max_dist = max([max([abs(r['offset_EW_arcsec'])/60, abs(r['offset_NS_arcsec'])/60]) for df in [stars_opt, stars_ir] if not isinstance(df, str) and not df.empty for _, r in df.head(3).iterrows()] + [0.0])
    
    # Set the final FOV: Must wrap the stars, but absolutely cannot be smaller than the instrument's minimum FOV
    dynamic_imsize = round(max(specs['min_fov'], (max_dist * 2) + 0.4 if max_dist > 0 else imsize), 1)

    # Download Optical and IR FITS images concurrently (asking for a 50% larger image to allow rotation cropping)
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_opt = executor.submit(get_image_fallbacks, ra, dec, s_name, dynamic_imsize*1.5)
        f_ir = executor.submit(get_image_2mass, ra, dec, s_name, dynamic_imsize*1.5)
        try: hdu_opt = f_opt.result()
        except: hdu_opt = None
        try: hdu_ir = f_ir.result()
        except: hdu_ir = None

    # Abort if absolutely no images could be downloaded
    if not hdu_opt and not hdu_ir: 
        raise ValueError("Could not fetch ANY images.")
            
    # Build the plot using the fetched data and dynamic constraints
    fig = fits2image_projected(hdu_opt, hdu_ir, stars_opt, stars_ir, pa_deg=pa_deg, imsize=dynamic_imsize, slit_width=slit_width, slit_height=specs['slit_h'], is_parallactic=is_parallactic)
    
    # Define save path
    base = Path(output_folder) / f"finder_{s_name}_PA{'PARA' if is_parallactic else pa_deg}.pdf"
    
    # Save the PDF file to disk
    fig.savefig(base, format="pdf", bbox_inches="tight", pad_inches=0.02)
    
    # Explicitly clear and close the Matplotlib figure to prevent massive memory leaks
    fig.clf() 
    plt.close('all') 

    logger.info(f"Saved: {base}")
    # If a Drive folder is specified, upload the PDF
    if drive_folder: 
        upload_to_drive(base, drive_folder)

# If executed from CLI, parse arguments and run
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ra", dest="ra_", required=True)
    args.add_argument("--dec", dest="dec_", required=True)
    args.add_argument("--s-name", default="Target")
    args.add_argument("--pa-deg", type=float, default=0.0)
    args.add_argument("--instrument", type=str, default="GOODMAN")
    args.add_argument("--output-folder", type=str, default='./finder_charts/')
    parsed = args.parse_args()
    run_pipeline(s_name=parsed.s_name, ra_str=parsed.ra_, dec_str=parsed.dec_, instrument=parsed.instrument, pa_deg=parsed.pa_deg, output_folder=parsed.output_folder)
