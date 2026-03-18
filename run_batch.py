# Import parsing, JSON, and time modules
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
# Import ProcessPoolExecutor for true parallel processing (multiprocessing)
from concurrent.futures import ProcessPoolExecutor

# Import the finder module (the plotting engine)
import finder
# Import the API fetcher
from soar_api import fetch_soar_data_to_json
# Import utility functions for Drive, logging, and cache management
from utils import check_file_in_drive, get_or_create_drive_folder, setup_logger, manage_cache_size

# Initialize the logger for the batch processor
logger = setup_logger(name="batch_processor")
# Define the local file used to remember which IDs have already been processed
STATE_FILE = "processed_ids.json"

def load_processed_ids():
    # If the state file exists, read it and return a set of processed IDs
    if Path(STATE_FILE).exists():
        with open(STATE_FILE, 'r') as f: 
            return set(json.load(f))
    # Otherwise, return an empty set
    return set()

def save_processed_ids(ids_set):
    # Save the current set of processed IDs back to the JSON file
    with open(STATE_FILE, 'w') as f: 
        json.dump(list(ids_set), f)

def parse_args():
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser()
    # Argument for the Master Google Drive Folder ID
    parser.add_argument("--drive-folder", type=str, default=None)
    # Argument for the local output directory
    parser.add_argument("--output-folder", type=str, default='./finder_charts/')
    # Argument to provide a local JSON file (bypassing the API)
    parser.add_argument("--input-json", type=str, default=None)
    # Flag to run the script exactly once and exit (no infinite loop)
    parser.add_argument("--run-once", action="store_true")
    # Argument to define how many CPU cores to use for parallel processing
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel processes")
    return parser.parse_args()

def process_single_target(obs, args, drive_folder_cache):
    # Extract details from the observation dictionary
    obs_id = obs.get('id')
    s_name = str(obs.get('object_name', 'Unknown')).replace(' ', '_')
    ra, dec = obs.get('ra'), obs.get('dec')
    start_time_str = obs.get('start_time')
    instrument = obs.get('instrument', 'GOODMAN')

    # Skip this target if critical data is missing
    if ra is None or dec is None or start_time_str is None: 
        return None

    try:
        # Clean the time string (remove T, Z, and milliseconds)
        clean_time_str = start_time_str.replace('T', ' ').replace('Z', '').split('+')[0].split('.')[0]
        # Subtract 12 hours to calculate the "Astronomical Night"
        night_date = (datetime.strptime(clean_time_str, '%Y-%m-%d %H:%M:%S') - timedelta(hours=12)).strftime('%Y-%m-%d')
        night_folder_name = f"Night_{night_date}"
    except Exception:
        # Fallback if time parsing fails
        night_folder_name = "Night_Unknown"
    
    # Create the local subfolder for this night (exist_ok=True prevents errors if it exists)
    local_night_folder = Path(args.output_folder) / night_folder_name
    local_night_folder.mkdir(parents=True, exist_ok=True)

    # Fetch the pre-created Drive folder ID from the cache dictionary
    current_drive_folder_id = drive_folder_cache.get(night_folder_name) if args.drive_folder else None

    # Determine Position Angle (PA) and if it requires Parallactic rotation
    raw_pa = obs.get('pa', 0.0)
    is_parallactic = str(raw_pa).lower() in ["para", "parallactic", "paralactico"]
    pa_value = 0.0 if is_parallactic else float(raw_pa)

    # Define what the final PDF filename should look like
    expected_filename = f"finder_{s_name}_PA{'PARA' if is_parallactic else pa_value}.pdf"
    
    # Check if the file already exists locally; skip if true
    if (local_night_folder / expected_filename).exists():
        logger.info(f"⏭️ Skipping '{s_name}': Exists locally.")
        return obs_id
    # Check if the file already exists in Google Drive; skip if true
    if current_drive_folder_id and check_file_in_drive(expected_filename, current_drive_folder_id):
        logger.info(f"⏭️ Skipping '{s_name}': Exists in Drive.")
        return obs_id

    # Log that generation is starting
    logger.info(f"Generating chart: {s_name} (Inst: {instrument}, Night: {night_folder_name})")
    try:
        # Execute the plotting pipeline from finder.py
        finder.run_pipeline(
            s_name=s_name, ra_str=str(ra), dec_str=str(dec), pa_deg=pa_value,
            instrument=instrument,
            imsize=float(obs.get('fov', 3.0)), radius=1.0, contrast=float(obs.get('contrast', 0.045)),
            slit_width=float(obs.get('slit', 1.0)), output_folder=str(local_night_folder),
            drive_folder=current_drive_folder_id, is_parallactic=is_parallactic 
        )
        # Return the ID on success
        return obs_id
    except Exception as e:
        # Log any errors that occurred during generation
        logger.error(f"❌ Failed to process {s_name}: {e}")
        return None

def process_batch(args):
    # Use the local JSON if provided, otherwise fetch fresh data from the API
    input_file = args.input_json if args.input_json else fetch_soar_data_to_json()
    # Abort if no file was found or generated
    if not input_file or not Path(input_file).exists(): return

    # Load the targets from the JSON file
    with open(input_file, 'r', encoding='utf-8') as f: 
        targets = json.load(f)

    # Load the IDs we have already processed in the past
    processed_ids = load_processed_ids()
    drive_folder_cache = {}
    
    # --- SYNCHRONOUS PRE-PROCESSING ---
    # Find all unique "Nights" in this batch to prevent Race Conditions
    unique_nights = set()
    for obs in targets:
        if obs.get('id') in processed_ids: continue
        start_time_str = obs.get('start_time')
        if not start_time_str: continue
        try:
            # Calculate night exactly as in process_single_target
            clean_time_str = start_time_str.replace('T', ' ').replace('Z', '').split('+')[0].split('.')[0]
            night_date = (datetime.strptime(clean_time_str, '%Y-%m-%d %H:%M:%S') - timedelta(hours=12)).strftime('%Y-%m-%d')
            unique_nights.add(f"Night_{night_date}")
        except Exception:
            unique_nights.add("Night_Unknown")
            
    # If Drive is enabled, synchronously create/fetch all required folders first
    if args.drive_folder:
        for night in unique_nights:
            drive_folder_cache[night] = get_or_create_drive_folder(night, args.drive_folder)
    
    # Synchronously clean up the local FITS cache before launching parallel workers
    logger.info("Verifying local FITS cache size before parallel processing...")
    manage_cache_size(cache_dir="./fits_cache", max_size_gb=3.0)
    # ----------------------------------

    # Launch the parallel processing pool
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        # Submit each unprocessed target to a worker process
        for obs in targets:
            if obs.get('id') in processed_ids: continue
            futures.append(executor.submit(process_single_target, obs, args, drive_folder_cache))
        
        # Collect the results as they finish
        for future in futures:
            success_id = future.result()
            # If successful, add the ID to the processed set
            if success_id: processed_ids.add(success_id)
            
    # Save the updated list of processed IDs to the local file
    save_processed_ids(processed_ids)

def main():
    # Parse terminal arguments
    args = parse_args()
    
    # If the user requested a single run, do it and exit
    if args.run_once:
        logger.info(f"Running single test cycle with {args.max_workers} parallel workers...")
        process_batch(args)
        return

    # Otherwise, enter an infinite loop (daemon mode)
    while True:
        logger.info(f"Starting review cycle: {time.strftime('%H:%M:%S')}")
        process_batch(args)
        # Sleep for 5 minutes (300 seconds) before checking the API again
        time.sleep(300)

if __name__ == "__main__":
    main()
