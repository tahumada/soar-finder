🔭 AEON/SOAR Automated Finder Chart Pipeline
An enterprise-grade, fully automated pipeline designed to generate astronomical Finder Charts for the SOAR Telescope and the broader AEON (Astronomical Event Observatory Network).

Built for 24/7 facility-level operation, this system continuously polls observation schedules, dynamically calculates optimal Fields of View (FOV) based on instrument specifications, queries multiple star catalogs with automatic failovers, and uploads memory-safe PDF renders directly to Google Drive.

✨ Core Features & Strengths
AEON Network & ToO Ready: Native support for dynamic instrument changes (e.g., SOAR Goodman, Gemini GMOS) and Target of Opportunity (ToO) alerts. It tracks observations by unique API id rather than coordinates, ensuring updated requests are never skipped.

Smart Dynamic FOV & Star Selection: Automatically scales the camera FOV to tightly fit the top 3 closest reference stars. It strictly enforces instrument-specific minimum FOVs and automatically excludes reference stars closer than 2.0" to the science target to prevent guiding on blended sources.

Extreme Network Resiliency: Implements exponential backoff (@retry_with_backoff) for unstable astronomical databases. If Gaia DR3 is down or lacks coverage, it gracefully falls back to Pan-STARRS, then to the Legacy Survey, ensuring a chart is always generated.

Multiprocessing Architecture: Utilizes Python's ProcessPoolExecutor to render multiple FITS images and PDFs simultaneously. This circumvents Python's GIL and Matplotlib's thread-locking issues, drastically reducing batch processing times.

Thread-Safe Google Drive Integration: Features a robust Singleton pattern for Google API credentials to avoid rate limits. It supports Team/Shared Drives and utilizes synchronous pre-processing to eliminate race conditions when creating dynamic Night_YYYY-MM-DD folders.

Memory-Safe Daemon Mode: Built to run infinitely. It uses Matplotlib's headless 'Agg' backend and forces explicit garbage collection (fig.clf(), plt.close('all')) to completely eliminate memory leaks during 24/7 server operations.

Intelligent Local Caching: Synchronously manages a local FITS file cache (capped at 3.0 GB by default) to prevent redundant downloads and save bandwidth.

⚙️ Prerequisites & Installation
1. Python Environment
Ensure you are using Python 3.9+ (3.10 or 3.11 recommended). Install the required dependencies:

Bash
pip install numpy matplotlib astropy astroquery pyvo requests python-dotenv google-api-python-client google-auth-httplib2 google-auth-oauthlib reproject charset-normalizer
2. AEON/LCO API Token (.env)
Create a file named .env in the root directory to securely store your LCO portal token:

Plaintext
SOAR_API_TOKEN=Token YOUR_SECRET_TOKEN_HERE
3. Google Drive Service Account
Place your Google Cloud Service Account JSON key in the root directory and name it exactly: drive_credentials.json. Ensure this service account has Editor access to your target Drive folder.

🚀 How to Use
1. Production Mode (24/7 Continuous Daemon)
This is the standard mode for observatory servers. The script will fetch the next 30 days of observations, process new targets, upload them, and then sleep for 5 minutes before checking the API again.

It is highly recommended to run this inside a tmux or screen session, or using nohup to keep it alive in the background:

Bash
# Start a tmux session
tmux new -s finder_pipeline

# Run the pipeline specifying your target Google Drive folder ID and CPU cores
python run_batch.py --drive-folder "YOUR_DRIVE_FOLDER_ID" --max-workers 4

# Detach from tmux: Press Ctrl+B, then D
2. Stress Test / Single Batch Mode
If you want to process a specific local JSON file (useful for testing edge-cases or historical data) without entering the infinite loop, use the --run-once and --input-json flags:

Bash
python run_batch.py --input-json test_observations.json --run-once --drive-folder "YOUR_DRIVE_FOLDER_ID" --max-workers 6
Note: If you want to force the pipeline to re-process targets it has already completed, delete the local processed_ids.json file.

3. Standalone Manual Chart Generation
If an astronomer needs a chart immediately on the fly, you can bypass the batch processor and call the core engine directly:

Bash
python finder.py --s-name "Supernova_2026A" --ra "183.053167" --dec "13.221750" --pa-deg 45.0 --instrument "GOODMAN"
📂 Project Architecture
The pipeline is strictly modularized into four key components:

run_batch.py: The Master Controller. Handles multiprocessing, calculates Astronomical Nights (T-12h), manages state (processed_ids.json), and coordinates synchronous tasks (cache cleaning, Drive folder creation) to prevent race conditions.

finder.py: The Core Plotting Engine. Calculates target-to-star radial distances, dynamically adjusts the FOV based on the INSTRUMENT_SPECS dictionary, and uses reproject and matplotlib to render the final PDF with WCS-accurate compass roses and slit overlays.

soar_api.py: The API Connector. Securely authenticates with the LCO proxy, queries a rolling 31-day window, and formats the raw schedule into a clean JSON digest.

utils.py: The Toolbelt. Contains the Astropy coordinate parsers, multithreaded FITS downloaders, Google Drive Singleton handlers, the 3GB local cache manager, and the centralized logging configuration.

📊 Logging & Monitoring
The pipeline is designed for headless servers and is entirely silent on stdout by default. All activities, warnings, API rate-limit delays, and successful uploads are safely recorded in aeon_pipeline.log.

To monitor the pipeline in real-time on a server:

Bash
tail -f aeon_pipeline.log
