import io
import os
import math
import re
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, ZScaleInterval

# Astroquery
from astroquery.mast import Catalogs
from astroquery.gaiagit  import Gaia
from astroquery.skyview import SkyView

# Reprojection
from reproject import reproject_interp

# defaults
mpl.rcParams["font.size"] = 15  # default = 10.0

def setup_logger(
    name="myapp",
    logfile="myapp.log",
    level=logging.INFO,
):
    log_path = Path(logfile)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a")  # append
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_coords(ra_str, dec_str):
    ra_str = ra_str.strip().lower()
    dec_str = dec_str.strip().lower()

    if ':' in ra_str or any(c in ra_str for c in 'hms'):
        ra = Angle(ra_str, unit=u.hourangle)
    else:
        ra = Angle(float(ra_str), unit=u.deg)
    
    if any(c in dec_str for c in 'dms') or ':' in dec_str:
        dec = Angle(dec_str, unit=u.deg)
    else:
        dec = Angle(float(dec_str), unit=u.deg)

    return ra.deg, dec.deg

def get_url(*args, **kwargs):
    # Connect and read timeouts
    kwargs["timeout"] = (6.05, 20)
    try:
        return requests.get(*args, **kwargs)
    except requests.exceptions.RequestException:
        return None

        
##### to query star catalogs

def query_stars_gaia(ra,dec,radius = 3):
    '''
    funtion to perform a cone search of stars from Gaia. Queality check on the photometry embedded in the query.
    input
        ra: RA in deg
        dec: Dec in deg
        radius: search radius in arcmin
    returns
        df: Pandas DataFrame with ra, dec, mag_g, and others (source_id, phot_bp_mean_mag, phot_rp_mean_mag, ruwe)
    '''
    
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    
    query = f"""
    SELECT
        source_id,
        ra,
        dec,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag,
        ruwe
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {(radius*u.arcmin).to(u.deg).value})
    )
    AND ruwe < 1.4
    AND visibility_periods_used > 8
    AND astrometric_excess_noise < 1
    AND phot_bp_rp_excess_factor BETWEEN
        1.0 + 0.015 * POWER(phot_bp_mean_mag - phot_rp_mean_mag, 2)
        AND
        1.3 + 0.06 * POWER(phot_bp_mean_mag - phot_rp_mean_mag, 2)
    ORDER BY phot_g_mean_mag ASC
    """
    
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    
    results
    df = results.to_pandas()
    df.keys()
    df = df.rename(columns={
        "phot_g_mean_mag": "mag",
            })

    return df

def query_stars_ps1(ra,dec,radius = 3):
    
    '''
    funtion to perform a cone search of stars from PS1. Queality check on the photometry embedded in the query.
    input
        ra: RA in deg
        dec: Dec in deg
        radius: search radius in arcmin
    returns
        df: Pandas DataFrame with ra, dec, mag (mag_g), mag_r and others (rKronMag,
qualityFlag)
    '''
    
    coord = SkyCoord(ra*u.deg,dec*u.deg)
    
    tbl = Catalogs.query_region(
        coord,
        radius=radius*u.arcmin,
        catalog="Panstarrs",
        table="stack",
        columns=["raMean", "decMean", "gPSFMag", "rPSFMag","rKronMag","qualityFlag"]
    )
    
    tbl = tbl[
        (tbl["rPSFMag"] < 19) 
        &
        (tbl["rPSFMag"] > 14) 
        &
        (abs(tbl["rPSFMag"] - tbl["rKronMag"]) < 0.05) 
        &
        (tbl["qualityFlag"] < 128)
    ]
    
    tbl.sort("rPSFMag")
    tbl[-1]['rPSFMag']
    df = tbl.to_pandas()
    df = df.rename(columns={
    "raMean": "ra",
    "decMean": "dec",
    "gPSFMag": "mag",
    "rPSFMag": "mag_r"
        })
    
    return df
    

import pyvo
import astropy.units as u
from astropy.coordinates import SkyCoord
import pyvo
from astropy.table import Table

def query_stars_ls(ra,dec,radius = 6):
    '''
    funtion to perform a box search of stars from Legacy Survey. Queality check on the photometry embedded in the query.
    input    
        ra: RA in deg
        dec: Dec in deg
        radius: size of the box in arcmin
    returns
        df: Pandas DataFrame with ra, dec, mag (mag_g), mag_r, mag_z
    '''
    
    
    # TAP service URL
    tap_url = "https://datalab.noirlab.edu/tap"
    
    # Create TAP service object
    tap_service = pyvo.dal.TAPService(tap_url)
    
    query = f"""
    SELECT TOP 100
        ra, dec,
        mag_g, mag_r, mag_z
    FROM ls_dr10.tractor
    WHERE
        type = 'PSF' 
        AND ra BETWEEN {ra-radius/2/60} AND {ra+radius/2/60}
        AND dec BETWEEN {dec-radius/2/60} AND {dec+radius/2/60}
        AND mag_r < 18
    """
    
    job = tap_service.run_async(query, language="ADQL")
    results = job.to_table()
    
    df = results.to_pandas()
    df = df.rename(columns={
        "mag_g": "mag",
            })
    
    
    return df

def get_stars_2mass(ra,dec,radius=2):

    from astroquery.irsa import Irsa
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    # Target position
    target = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
        
    # Query 2MASS Point Source Catalog
    tbl = Irsa.query_region(
        target,
        radius=radius* u.arcmin,
        catalog="fp_psc"   # 2MASS Point Source Catalog
    )
    
    # Extract J-band photometry
    tbl = tbl["ra", "dec", "j_m", "j_cmsig", "ph_qual", "cc_flg"]
    
    good = np.array([ph_qa[0] =='A' or ph_qa[0] =='B' for ph_qa in tbl["ph_qual"]]) #* np.array([ph_qa[0] =='0' for ph_qa in tbl["cc_flg"]])    
    
    clean = tbl[good]
    
    df = clean.to_pandas()
    df = df.rename(columns={
        "j_m": "mag",
            })
    
    
    return df


def get_image_ps1(ra,dec,source_name,imsize=6):
    
    url = (
            f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
            f"?width=500&height=500&fov={imsize/60}&ra={ra}&dec={dec}"
            f"&hips=CDS/P/PanSTARRS/DR1/r"
        )
    
    response = requests.get(url, timeout=30)
    hdu = fits.open(BytesIO(response.content))
    
    npixels = len(hdu[0].data)
    pixscale = imsize*60/npixels
    
    if response is None or response.status_code != 200:
            print('failed PS1 retrieval')
            return ''
        
        
    hdu = fits.open(BytesIO(response.content))
    if len(hdu) == 0:
        print('failed PS1 retrieval')
        return ''
        
    hdu[0].header['watermark'] =  'PS1'
    hdu[0].header['pixscale'] =  pixscale
    hdu[0].header['imsize'] =  imsize
    hdu[0].header['source_name'] =  source_name
    hdu[0].header['ra'] =  ra
    hdu[0].header['dec'] =  dec
    hdu[0].header['numpix'] = npixels

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
    if len(hdu) == 0:
        print('failed')
        return ''
        
    hdu[0].header['watermark'] =  'LS'
    hdu[0].header['pixscale'] =  pixscale
    hdu[0].header['imsize'] =  imsize
    hdu[0].header['source_name'] =  source_name
    hdu[0].header['ra'] =  ra
    hdu[0].header['dec'] =  dec
    hdu[0].header['numpix'] =  numpix

    return hdu

def get_image_decaps(ra,dec,source_name,
                  imsize = 6# in arcmin
                 ):
    
    pixscale = 0.26
    DECAPS_CUTOUT_TIMEOUT = 30
    
    numpix = math.ceil(60 * imsize / pixscale)
    
    decaps_query_url = (
         f"http://legacysurvey.org/viewer/fits-cutout/?layer=decaps2&"
         f"ra={ra}&dec={dec}&pixscale={pixscale}&bands=r&size={numpix}"
    )
    print(decaps_query_url)
    
    response = requests.get(decaps_query_url, timeout=DECAPS_CUTOUT_TIMEOUT)
    if response is None or response.status_code != 200:
        print('failed')
        return ''
    
    
    hdu = fits.open(BytesIO(response.content))
    if len(hdu) == 0:
        print('failed')
        return ''
        
    hdu[0].header['watermark'] =  'LS'
    hdu[0].header['pixscale'] =  pixscale
    hdu[0].header['imsize'] =  imsize
    hdu[0].header['source_name'] =  source_name
    hdu[0].header['ra'] =  ra
    hdu[0].header['dec'] =  dec
    hdu[0].header['numpix'] =  numpix

    return hdu

def get_image_dss(ra,dec,source_name,
                  imsize = 6# in arcmin
                 ):
    
    url = f"http://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red&r={ra}&dec={dec}&h={imsize}&w={imsize}&e=J2000"
    response = requests.get(url, timeout=30)
    hdu = fits.open(BytesIO(response.content))
    
    npixels = len(hdu[0].data)
    imsize = imsize
    pixscale = imsize*60/npixels
    
    if response is None or response.status_code != 200:
            print('failed DSS retrieval')
            return ''
        
        
    hdu = fits.open(BytesIO(response.content))
    if len(hdu) == 0:
        print('failed DSS retrieval')
        return ''
        
    hdu[0].header['watermark'] =  'DSS'
    hdu[0].header['pixscale'] =  pixscale
    hdu[0].header['imsize'] =  imsize
    hdu[0].header['source_name'] =  source_name
    hdu[0].header['ra'] =  ra
    hdu[0].header['dec'] =  dec
    hdu[0].header['numpix'] =  len(hdu[0].data)

    return hdu

def get_image_2mass(ra,dec,source_name,
                  imsize = 6# in arcmin
                 ):
    
    url = (
    "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?"
    f"Position={ra},{dec}"
    "&Survey=2MASS-J"
    f"&Radius={imsize/60}"   # degrees = 5 arcmin
    "&Return=FITS"
    )

    response = requests.get(url, timeout=30)
    hdu = fits.open(BytesIO(response.content))
    
    npixels = len(hdu[0].data)
    pixscale = imsize*60/npixels
    
    if response is None or response.status_code != 200:
            print('failed 2MASS retrieval')
            return ''
        
        
    hdu = fits.open(BytesIO(response.content))
    im = hdu[0].data
    cent = int(npixels / 2)
    width = int(0.05 * npixels)
    test_slice = slice(cent - width, cent + width)
    all_nans = np.isnan(im[test_slice, test_slice].flatten()).all()
    all_zeros = (im[test_slice, test_slice].flatten() == 0).all()
  
    if len(hdu) == 0 or all_zeros or all_nans:
        print('failed 2MASS retrieval, trying again')
        
        coord = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
        images = SkyView.get_images(
            position=coord,
            survey=["2MASS-J"],   # 2MASS-J, 2MASS-H, 2MASS-K
            radius=imsize * u.arcmin,
            pixels=npixels
        )
        hdu = images[0]
        im = hdu[0].data
        all_nans = np.isnan(im[test_slice, test_slice].flatten()).all()
        all_zeros = (im[test_slice, test_slice].flatten() == 0).all()
        if len(im) == 0 or all_zeros or all_nans:
            return ''
        
    hdu[0].header['watermark'] =  '2MASS'
    hdu[0].header['pixscale'] =  pixscale
    hdu[0].header['imsize'] =  imsize
    hdu[0].header['source_name'] =  source_name
    hdu[0].header['ra'] =  ra
    hdu[0].header['dec'] =  dec
    hdu[0].header['numpix'] = npixels

    return hdu

def get_image_fallbacks(ra,dec,source_name,imsize = 5):
    hdu = get_image_ls(ra,dec,source_name,imsize = imsize)
    if hdu != '':
        print('image from LS')
    
    if hdu == '':
        hdu = get_image_ps1(ra,dec,source_name,imsize = imsize)
        if hdu != '':
            print('image from PS1')
        
    if hdu == '':
        hdu = get_image_decaps(ra,dec,source_name,imsize = imsize)
        if hdu != '':
            print('image from DECaPs')

    if hdu == '':
        hdu = get_image_dss(ra,dec,source_name,imsize = imsize)
        if hdu != '':
            print('image from DSS')
    
    if hdu == '':
        raise TypeError("could not get image, tried LS, PS1, DECaPs, and DSS")
    
    return hdu