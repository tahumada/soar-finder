import argparse
from pathlib import Path
# Import the finder module directly to avoid subprocess overhead
import finder

def parse_args():
    # Set up argument parser for the batch script
    parser = argparse.ArgumentParser(description="Run finder pipeline for a list of objects.")
    # Define the required input file argument
    parser.add_argument("input_file", help="Input .txt file with targets (Columns: Name RA DEC [pa=X] [fov=Y] [contrast=Z])")
    # Optionally pass the Google Drive folder ID to upload all generated PDFs
    parser.add_argument("--drive-folder", type=str, default=None, help="Google Drive Folder ID")
    return parser.parse_args()

def main():
    args = parse_args()

    # Check if the input text file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Could not find the file '{args.input_file}'.")
        return

    # Open and read the file line by line, removing the hidden BOM character
    with open(input_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            # Strip leading and trailing whitespaces
            line = line.strip()
            
            # Skip empty lines or comments
            if not line or line.startswith('#'):
                continue
            
            # Split the line into columns based on spaces or tabs
            parts = line.split()
            
            # Ensure there are at least 3 columns (Name, RA, DEC)
            if len(parts) < 3:
                print(f"Skipping invalid line (missing columns): {line}")
                continue
            
            # Assign primary columns to respective variables
            s_name = parts[0]
            ra = parts[1]
            dec = parts[2]
            
            # Define default optional values
            pa_value = 0.0
            imsize_value = 4.0
            contrast_value = 0.045
            
            # Extract optional arguments from remaining columns
            if len(parts) > 3:
                for part in parts[3:]:
                    part_lower = part.lower()
                    if part_lower.startswith("pa="):
                        pa_value = float(part.split("=")[1])
                    elif part_lower.startswith("fov="):
                        imsize_value = float(part.split("=")[1])
                    elif part_lower.startswith("contrast="):
                        contrast_value = float(part.split("=")[1])
            
            # Print status message to console
            print(f"\n{'='*65}")
            print(f"Processing target: {s_name} (RA: {ra}, DEC: {dec})")
            print(f"Options | PA: {pa_value}° | FOV: {imsize_value}' | Contrast: {contrast_value}")
            print(f"{'='*65}")
            
            # Run the pipeline synchronously as a direct Python function call
            try:
                finder.run_pipeline(
                    s_name=s_name,
                    ra_str=ra,
                    dec_str=dec,
                    pa_deg=pa_value,
                    imsize=imsize_value,
                    radius=1.0, # Default star search radius
                    contrast=contrast_value,
                    output_folder='./finder_charts/',
                    drive_folder=args.drive_folder,
                )
            except Exception as e:
                print(f"❌ Failed to process {s_name} completely: {e}")

if __name__ == "__main__":
    main()
