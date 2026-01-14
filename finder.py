# todo - the finder when applied PA is turned the other way around

# Standard library
import io
import os
import math
import re
from io import BytesIO
from pathlib import Path

# Third-party
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests

# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, ZScaleInterval

# Reproject
from reproject import reproject_interp

# utils
from utils import query_stars_gaia, query_stars_ps1, query_stars_ls, get_url, parse_coords


# Matplotlib defaults
mpl.rcParams["font.size"] = 15  # default = 10.0

def get_image_ps1(ra,dec,source_name,imsize=6):
    
    PS1_CUTOUT_TIMEOUT = 30
    pixscale = 0.258

    numpix = int(round(60 * imsize / pixscale))
    if numpix % 2 == 0:
        numpix += 1

    # numpix = math.ceil(60 * imsize / pixscale)
    
    ps_query_url = (
        f"http://ps1images.stsci.edu/cgi-bin/ps1cutouts"
        f"?pos={ra:.8f}+{dec:.8f}&"
        f"filter=r&filetypes=stack&size={numpix}"
    )
    
    try:
        response = requests.get(ps_query_url, timeout=PS1_CUTOUT_TIMEOUT)
        # see models.py for how this URL is constructed
        match = re.search('src="//ps1images.stsci.edu.*?"', response.content.decode())
        if match is None:
            print(f"PS1 image not found for {ra} {dec}")
            return ""
        url = match.group().replace('src="', "http:").replace('"', "")
        url += f"&format=fits&imagename=ps1{ra}{dec:+f}.fits"
    except (requests.exceptions.SSLError, requests.exceptions.ReadTimeout) as e:
        print(f"Error getting PS1 image URL {str(e)}")
        return ""
    except Exception as e:
        print(f"Error getting PS1 image URL {e.message}")
        return ""
    
    response = get_url(url, stream=True, allow_redirects=True)
    
    if response is None or response.status_code != 200:
        print('failed')
    
    # Check if HDU is a valid FITS file
    hdu = fits.open(io.BytesIO(response.content))
   
    hdu[0].header['watermark'] =  'PS1'
    hdu[0].header['pixscale'] =  pixscale
    hdu[0].header['imsize'] =  imsize
    hdu[0].header['source_name'] =  source_name
    hdu[0].header['ra'] =  ra
    hdu[0].header['dec'] =  dec
    hdu[0].header['numpix'] =  numpix
    
    return hdu


def get_image_ls(ra,dec,source_name,
                  imsize = 6# in arcmin
                 ):
    
    pixscale = 0.26
    LS_CUTOUT_TIMEOUT = 30
    
    numpix = math.ceil(60 * imsize / pixscale)
    
    ls_query_url = (
         f"http://legacysurvey.org/viewer/fits-cutout/"
         f"?ra={ra}&dec={dec}&layer=dr8&pixscale={pixscale}&bands=r&size={numpix}"
    )
    print(ls_query_url)
    
    response = requests.get(ls_query_url, timeout=LS_CUTOUT_TIMEOUT)
    if response is None or response.status_code != 200:
        print('failed')
        return ''
    
    
    hdu = fits.open(BytesIO(response.content))
    hdu[0].header['watermark'] =  'LS'
    hdu[0].header['pixscale'] =  pixscale
    hdu[0].header['imsize'] =  imsize
    hdu[0].header['source_name'] =  source_name
    hdu[0].header['ra'] =  ra
    hdu[0].header['dec'] =  dec
    hdu[0].header['numpix'] =  numpix

    return hdu


def get_stars (ra,dec,radius=3):

    stars = query_stars_gaia(ra,dec,radius=radius)
    if len(stars) != 0:
        print('Star catalog from Gaia')
    if len(stars) == 0 and dec > -30:
        stars = query_stars_ps1(ra,dec,radius=radius)
        if len(stars) != 0:
            print('Star catalog from PS1')
    if len(stars) == 0 and dec < 30:
        stars = query_stars_ls(ra,dec,radius=radius)
        if len(stars) != 0:
            print('Star catalog from LS')
    if len(stars) == 0:
        print('Failed to retrieve stars')
        return ''
        
    # add offset columns

    target = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
    coords = SkyCoord(stars["ra"].values*u.deg,
                       stars["dec"].values*u.deg,
                       frame="icrs")
    
    # Offsets: dra = East-West, ddec = North-South
    dra, ddec = coords.spherical_offsets_to(target)
    stars["offset_EW_arcsec"] = dra.to(u.arcsec).value
    stars["offset_NS_arcsec"] = ddec.to(u.arcsec).value

    return stars

def fits2image_projected(hdu,pa_deg=0,imsize = 6,radius = 3):

    watermark = hdu[0].header['watermark']
    imsize = hdu[0].header['imsize']
    pixscale = hdu[0].header['pixscale']
    source_name = hdu[0].header['source_name']
    ra = hdu[0].header['ra']
    dec = hdu[0].header['dec']
    npixels = hdu[0].header['numpix']
    
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=False)
    widths = [2.6, 1]
    heights = [2.6, 1]
    spec = fig.add_gridspec(
        ncols=2,
        nrows=1,
        width_ratios=widths,
        # height_ratios=heights,
        left=0.05,
        right=0.95,
    )
    
    # npixels = 60 * imsize / pixscale #= source_image_parameters[image_source].get("npixels", 256)
    # pixscale = 60 * imsize / numpix
    
    wcs = WCS(naxis=2)
    # set the headers of the WCS.
    # The center of the image is the reference point (source_ra, source_dec):
    wcs.wcs.crpix = [npixels / 2, npixels / 2]
    wcs.wcs.crval = [ra, dec]
    
    # create the pixel scale and orientation North up, East left
    # pixelscale is in degrees, established in the tangent plane
    # to the reference point
    
    wcs.wcs.cd = np.array([[-pixscale / 3600, 0], [0, pixscale / 3600]])
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    im = hdu[0].data
    ######## rotate
    if pa_deg != 0:
        pa_rad = np.deg2rad(360-pa_deg)
        
        # Copy WCS
        new_wcs = wcs.deepcopy()
        
        # Rotation matrix (PC matrix)
        rot = np.array([
            [ np.cos(pa_rad), -np.sin(pa_rad)],
            [ np.sin(pa_rad),  np.cos(pa_rad)]
        ])
        
        new_wcs.wcs.cd = rot @ wcs.wcs.cd
        
        reproj_data, footprint = reproject_interp(
            (hdu[0].data, wcs),
            new_wcs,
            shape_out=hdu[0].data.shape
        )
        
        im = reproj_data
        
    else:
        new_wcs = wcs.deepcopy()
    
    # replace the nans with medians
    im[np.isnan(im)] = np.nanmedian(im)
    
    # Fix the header keyword for the input system, if needed
    hdr = hdu[0].header
    if "RADECSYS" in hdr:
        hdr.set("RADESYSa", hdr["RADECSYS"], before="RADECSYS")
        del hdr["RADECSYS"]
    
    wcs = WCS(hdu[0].header)
    
    zscale_contrast=0.045
    zscale_krej=2.5
    
    cent = int(npixels / 2)
    width = int(0.05 * npixels)
    test_slice = slice(cent - width, cent + width)
    all_nans = np.isnan(im[test_slice, test_slice].flatten()).all()
    all_zeros = (im[test_slice, test_slice].flatten() == 0).all()
    if not (all_zeros or all_nans):
        percents = np.nanpercentile(im.flatten(), [10, 99.0])
        vmin = percents[0]
        vmax = percents[1]
        interval = ZScaleInterval(
            nsamples=int(0.1 * (im.shape[0] * im.shape[1])),
            contrast=zscale_contrast,
            krej=zscale_krej,
        )
        norm = ImageNormalize(im, vmin=vmin, vmax=vmax, interval=interval)
        watermark = watermark #source_image_parameters[image_source]["str"]
        fallback = False
    
    # add the images in the top left corner
    ax = fig.add_subplot(spec[0, 0], projection=new_wcs)
    ax_text = fig.add_subplot(spec[0, 1])
    ax_text.axis("off")
    # ax_starlist = fig.add_subplot(spec[1, 0:])
    # ax_starlist.axis("off")
    
    ax.imshow(im, origin="lower", norm=norm, cmap="gray_r")
    ax.set_autoscale_on(False)
    ax.grid(color="white", ls="dotted")
    ax.set_xlabel(r"$\alpha$ (J2000)", fontsize="large")
    ax.set_ylabel(r"$\delta$ (J2000)", fontsize="large")
    ax.set_title(
    f"{source_name} Finder",
    fontsize="large",
    fontweight="bold",
    )

    
    # plot hair cross for the target

    coord = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
    ra_str  = coord.ra.to_string(unit=u.hour, sep=":", precision=2)
    dec_str = coord.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True)

    x, y = new_wcs.world_to_pixel(coord)
    color_target = "#D55E00"
    arm_len = 30     # length of each arm (pixels)
    gap = 15          # half-gap at center (pixels)
    lw = 4
    mag = "?"
    offset_ra, offset_dec = 0,0

    star_text = (
            rf"$\bf{{{source_name}}} - {mag}$ mag"
            f'\n{coord.ra.deg:.6f} {coord.dec.deg:.6f}\n'
            f'{ra_str} {dec_str} \n'
            f'{round(offset_ra,4) } {round(offset_dec,4)}'
           )
    
    ax.plot([x + gap, x + arm_len], [y, y], color=color_target, lw=lw)
    ax.plot([x, x], [y + gap, y + arm_len], color=color_target, lw=lw)
    ax_text.text(0,
                 1 - (0.5) / 4,
                 star_text,
                color = color_target)
    
    # getting stars
    # radius = 3
    stars = get_stars(ra,dec,radius=radius)
    top3 = stars.iloc[:3]
    
    colors = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    ]  # any colors you want

    
    top3_colors = {}
    

    arm_len = 30     # length of each arm (pixels)
    gap = 15          # half-gap at center (pixels)
    lw = 3


    for i, (idx, row) in enumerate(top3.iterrows()):
        coord = SkyCoord(row.ra*u.deg, row.dec*u.deg, frame="icrs")
        x, y = new_wcs.world_to_pixel(coord)        
        ra_str  = coord.ra.to_string(unit=u.hour, sep=":", precision=2)
        dec_str = coord.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True)

        mag = row.mag_g
        offset_ra, offset_dec = row.offset_EW_arcsec,row.offset_NS_arcsec

        c = colors[i]
        top3_colors[idx] = c
        
        star_text = (
                rf"$\bf{{{source_name}-{str(i+1)}}}$  {mag:.2f} mag"
                f'\n{coord.ra.deg:.6f} {coord.dec.deg:.6f}\n'
                f'{ra_str} {dec_str} \n'
                f'offset RA, Dec: {round(offset_ra,4) } {round(offset_dec,4)}'
                f'\nRA > 0 East, Dec > 0 N'
               )
    
        c = colors[i]
        top3_colors[idx] = c
    
        # Horizontal arms
        ax.plot([x - arm_len, x - gap], [y, y], color=c, lw=lw)
        ax.plot([x + gap, x + arm_len], [y, y], color=c, lw=lw)
    
        # Vertical arms
        ax.plot([x, x], [y - arm_len, y - gap], color=c, lw=lw)
        ax.plot([x, x], [y + gap, y + arm_len], color=c, lw=lw)
    
    # add text on the side
        ax_text.text(0,
                     1 - ((i+1) + 0.5) / 4,
                     star_text,
                     color = c)
        print(star_text)

    
    # compass rose

    theta = np.deg2rad(360-pa_deg)
    length = 60        # arm length in pixels
    lw = 2.5
    gold = "#E69F00"
    x0, y0 = int(npixels * 0.1), int(npixels * 0.9)   # choose where to draw the rose

    dx_N =  length * np.sin(theta)
    dy_N =  length * np.cos(theta)
    
    ax.plot([x0, x0 + dx_N], [y0, y0 + dy_N],
            color=gold, lw=lw)
    ax.text(x0 + dx_N*1.4, y0 + dy_N*1.4, "E",
            color=gold, ha="center", va="center",rotation=pa_deg )

    dx_E =  length * np.cos(theta)
    dy_E = -length * np.sin(theta)
    
    ax.plot([x0, x0 + dx_E], [y0, y0 + dy_E],
            color=gold, lw=lw)
    ax.text(x0 + dx_E*1.4, y0 + dy_E*1.4, "N",
            color=gold, ha="center", va="center",rotation=pa_deg)
    
    return fig

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Make a finder for an object")

    # Required sky position
    parser.add_argument( "--ra", dest="ra_", required=True, help="Right Ascension (degree or sexagesimal, --ra=03:53:49.46)")
    parser.add_argument( "--dec", dest="dec_", required=True, help="Declination (degree or sexagesimal, e.g. --dec=-11:32:57.07)")

    # Optional parameters
    parser.add_argument( "--source-name", default="", help="Source name (string)")
    parser.add_argument( "--pa-deg", type=float, default=0.0, help="Position angle in degrees (East of North)")
    parser.add_argument( "--imsize", type=float, default=4.0, help="Image size in arcminutes")
    parser.add_argument("--radius", type=float, default=1.0, help="Radius in arcminutes")
    parser.add_argument("--output-folder", type=str, default='./finder_charts/', help="output folder")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # from the script
    ra_ = args.ra_
    dec_ = args.dec_
    
    # to add as variables
    source_name = args.source_name
    pa_deg = args.pa_deg
    imsize = args.imsize
    radius = args.radius
    output_folder = args.output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    
    # convert to astropy coords
    ra,dec = parse_coords(ra_,dec_)
    
    # Example usage
    print(f"Source: {args.source_name}")
    print(f"RA, Dec (deg): {ra:.6f}, {dec:.6f}")
    print(f"PA: {args.pa_deg} deg")
    print(f"Image size: {args.imsize} arcmin")
    print(f"Radius: {args.radius} arcmin")
    print(f"Output folder: {args.output_folder} ")
   
    output_file = output_folder + f'finder_{source_name}.pdf'
    print(f"Output file: {output_file} ")
    
    
    
    
    # download image
    
    hdu = get_image_ls(ra,dec,source_name,imsize = imsize)
    
    if hdu == '':
        print('image from PS1 a bit distorted, offsets are ok')
        hdu = get_image_ps1(ra,dec,source_name,imsize = imsize)
    
    fig = fits2image_projected(hdu,pa_deg=pa_deg, imsize = imsize,radius = radius)
    
    fig.savefig(
    output_file,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.02
    )

    return


if __name__ == "__main__":
    main()
    
    # test
    # python finder.py --source-name ZTF18aafhmpq --ra=111.12315 --dec=8.56597
# python finder.py --source-name ZTF25acjfaoq --ra 00:23:08.83 --dec=-24:40:51.43


