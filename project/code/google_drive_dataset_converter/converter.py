#!/usr/bin/env python3

import threading
import argparse
import queue
import piexif
import os
import sys
import re
import uuid
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from PIL import Image
import pillow_heif 

# Register HEIF/HEIC opener with PIL
pillow_heif.register_heif_opener()

# Supported image file extensions
PHOTO_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.heic', '.heif', '.bmp', '.tiff', '.webp'}
DIR_MIME = 'application/vnd.google-apps.folder'  # Google Drive folder mime type

def parse_args():
    """Parse command line arguments for the script.
    
    Returns:
        Namespace: Contains source_id, output_dir, last_date, and dest_id arguments
    """
    parser = argparse.ArgumentParser(description='Download and convert to .jpg (optionally upload again) Google Drive photos')
    parser.add_argument('--source-id', required=True, type=str, help='Google Drive source folder ID')
    parser.add_argument('--output-dir', default='dataset', type=str, help='Output directory (default: dataset)')
    parser.add_argument('--last-date', type=str, help='Only process files created after this date (DD-MM-YYYY)')
    parser.add_argument('--dest-id', type=str, help='Destination folder ID in Google Drive (enables cloud save)')
    return parser.parse_args()

def init_drive():
    """Initialize and authenticate Google Drive API connection.
    
    Returns:
        GoogleDrive: Authenticated Google Drive instance
    """
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # This will open a browser for authentication
    return GoogleDrive(gauth)

def extract_species_from_path(drive_path):
    """Extract species information from Google Drive path pattern.
    
    The path should follow the pattern: female(species1+species2) male(species3+species4)
    Both female and male parts are optional.
    
    Args:
        drive_path (str): Full path from Google Drive containing species info
        
    Returns:
        list: Sorted list of unique species found in the path
    """
    female_pattern = r"female\((.*?)\)"
    male_pattern = r"male\((.*?)\)"
    
    female_match = re.search(female_pattern, drive_path)
    male_match = re.search(male_pattern, drive_path)
    
    species_set = set()
    
    if female_match:
        female_species = female_match.group(1).split('+')
        species_set.update(female_species)
    
    if male_match:
        male_species = male_match.group(1).split('+')
        species_set.update(male_species)
    
    return sorted(list(species_set))

def generate_short_uuid():
    """Generate a short unique identifier (first 8 chars of UUID).
    
    Returns:
        str: First 8 characters of a UUID4
    """
    return str(uuid.uuid4()).split('-')[0]

def convert_to_jpg(file_path, drive_path=""):
    """Convert an image file to JPEG format with standardized naming.
    
    The new filename incorporates species information from the path and a UUID.
    Strips all EXIF metadata from the image.
    
    Args:
        file_path (str): Local path to the image file
        drive_path (str): Original Google Drive path (for species extraction)
        
    Returns:
        str: Path to the converted JPEG file, or None if conversion failed
    """
    try:
        # Extract species information for the new filename
        species_list = extract_species_from_path(drive_path)
        original_filename = os.path.basename(file_path)

        if not species_list:
            # Remove the Google Drive ID suffix we added during download
            clean_name = original_filename.split('_')[:-1]
            clean_name = '_'.join(clean_name) if clean_name else original_filename

            # If no species, keep original name but add UUID
            new_filename = os.path.splitext(clean_name)[0] + "_" + generate_short_uuid() + ".jpg"
        else:
            # Create standardized filename with species and UUID
            species_str = "-".join(species_list)
            new_filename = f"{species_str}_{generate_short_uuid()}.jpg"
        
        new_path = os.path.join(os.path.dirname(file_path), new_filename)
        
        # Convert image to JPEG and strip EXIF data
        with Image.open(file_path) as img:
            img_no_exif = Image.new(img.mode, img.size)
            img_no_exif.putdata(list(img.getdata()))
            img_no_exif.save(new_path, "JPEG", quality=95, subsampling=0)

        print(f"Converted: {original_filename} -> {os.path.basename(new_path)}")
        os.remove(file_path)
        return new_path
    except Exception as e:
        print(f"Error converting {os.path.basename(file_path)}: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return None

def converter_worker(file_q, upload_q=None):
    """Worker thread that processes files from queue and converts them.
    
    Args:
        file_q (Queue): Queue of (file_path, drive_path) tuples to process
        upload_q (Queue, optional): Queue to put converted files for uploading
    """
    while True:
        item = file_q.get()
        if item == "STOP":  # Termination signal
            break
        
        file_path, drive_path = item
        converted_path = convert_to_jpg(file_path, drive_path)
        if upload_q is not None and converted_path:
            upload_q.put(converted_path)
            
        file_q.task_done()

def uploader_worker(drive, dest_id, upload_q):
    """Worker thread that uploads converted files to Google Drive.
    
    Creates date-based folders in destination and maintains structure.
    
    Args:
        drive (GoogleDrive): Authenticated Google Drive instance
        dest_id (str): Destination folder ID in Google Drive
        upload_q (Queue): Queue of file paths to upload
    """
    while True:
        file_path = upload_q.get()
        if file_path == "STOP":  # Termination signal
            break
        
        try:
            date_dir = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)
            
            # Find or create date folder in destination
            query = f"'{dest_id}' in parents and title='{date_dir}' and mimeType='{DIR_MIME}' and trashed=false"
            date_folders = drive.ListFile({'q': query}).GetList()
            
            if not date_folders:
                date_folder = drive.CreateFile({
                    'title': date_dir,
                    'mimeType': DIR_MIME,
                    'parents': [{'id': dest_id}]
                })
                date_folder.Upload()
                print(f"Created remote folder: {date_dir}")
            else:
                date_folder = date_folders[0]
            
            # Upload the file to the date folder
            remote_file = drive.CreateFile({
                'title': file_name,
                'parents': [{'id': date_folder['id']}]
            })
            remote_file.SetContentFile(file_path)
            remote_file.Upload()
            print(f"Uploaded: {file_name} to {date_dir}")
            
        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")
        
        upload_q.task_done()

def parse_date(date_str):
    """Parse date string in DD-MM-YYYY format to datetime object.
    
    Args:
        date_str (str): Date string in DD-MM-YYYY format
        
    Returns:
        datetime: Parsed datetime object
    """
    return datetime.strptime(date_str, "%d-%m-%Y")

def get_date_dir_name(item):
    """Extract creation date from Google Drive file metadata and format it.
    
    Args:
        item (GoogleDriveFile): File metadata from Google Drive
        
    Returns:
        str: Date string in DD-MM-YYYY format
    """
    created_date = item['createdDate']
    dt = datetime.strptime(created_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    return dt.strftime("%d-%m-%Y")

def should_process_file(item, last_date_filter=None):
    """Check if a file should be processed based on creation date filter.
    
    Args:
        item (GoogleDriveFile): File metadata from Google Drive
        last_date_filter (datetime, optional): Minimum creation date to process
        
    Returns:
        bool: True if file should be processed, False otherwise
    """
    if not last_date_filter:
        return True
    
    created_date = datetime.strptime(item['createdDate'], "%Y-%m-%dT%H:%M:%S.%fZ")
    return created_date > last_date_filter

def get_drive_path(drive, file_id):
    """Get the full path of a file in Google Drive by walking up parent folders.
    
    Args:
        drive (GoogleDrive): Authenticated Google Drive instance
        file_id (str): ID of the file to get path for
        
    Returns:
        str: Full path string separated by slashes
    """
    path_parts = []
    current_id = file_id
    
    while current_id:
        try:
            file_info = drive.CreateFile({'id': current_id})
            file_info.FetchMetadata()
            
            path_parts.insert(0, file_info['title'])
            
            # Move to parent folder
            if 'parents' in file_info and file_info['parents']:
                current_id = file_info['parents'][0]['id']
            else:
                current_id = None
        except Exception as e:
            print(f"Error fetching path: {str(e)}")
            break
    
    return "/".join(path_parts)

def download_dir(drive, source_dir_id, local_dir, file_q, last_date_filter=None, parent_path=""):
    """Recursively download files from Google Drive folder to local directory.
    
    Args:
        drive (GoogleDrive): Authenticated Google Drive instance
        source_dir_id (str): ID of the Google Drive folder to download from
        local_dir (str): Local directory to save files to
        file_q (Queue): Queue to put downloaded files for processing
        last_date_filter (datetime, optional): Filter files by creation date
        parent_path (str, optional): Accumulated path from parent folders
    """
    file_list = drive.ListFile({'q': f"'{source_dir_id}' in parents and trashed=false"}).GetList()
    
    # Get current folder name if not the root source folder
    current_folder = ""
    if source_dir_id != parse_args().source_id:
        folder_info = drive.CreateFile({'id': source_dir_id})
        folder_info.FetchMetadata()
        current_folder = folder_info['title']
    
    # Build current path for species extraction
    current_path = parent_path
    if current_folder:
        current_path = f"{parent_path}/{current_folder}" if parent_path else current_folder
    
    for item in file_list:
        if item['mimeType'] == DIR_MIME:
            # Recursively process subdirectories
            download_dir(drive, item['id'], local_dir, file_q, last_date_filter, current_path)
        else:
            file_ext = os.path.splitext(item['title'])[1].lower()
            if file_ext in PHOTO_EXTENSIONS and should_process_file(item, last_date_filter):
                date_dir = get_date_dir_name(item)
                target_dir = os.path.join(local_dir, date_dir)
                os.makedirs(target_dir, exist_ok=True)
                
                # Build full Drive path for this file for species extraction
                file_drive_path = f"{current_path}/{item['title']}" if current_path else item['title']
                
                # Generate unique filenames to do not loose files that are not duplicates, because overwriting
                file_id = item['id']
                original_name = os.path.splitext(item['title'])[0]
                unique_name = f"{original_name}_{file_id[:8]}{file_ext}"
                target_path = os.path.join(target_dir, unique_name)

                if not os.path.exists(target_path):
                    print(f"Downloading: {item['title']} as {unique_name}")
                    try:
                        item.GetContentFile(target_path)
                        file_q.put((target_path, file_drive_path))
                    except Exception as e:
                        print(f"Error downloading {item['title']}: {str(e)}")

def main():
    """Main function that coordinates the download, conversion and upload process."""
    args = parse_args()
    
    # Parse date filter if provided
    last_date_filter = None
    if args.last_date:
        try:
            last_date_filter = parse_date(args.last_date)
            print(f"Processing files created after: {last_date_filter.strftime('%d-%m-%Y')}")
        except ValueError:
            print("Invalid date format. Please use DD-MM-YYYY", file=sys.stderr)
            sys.exit(1)

    # Setup queues and worker threads
    file_q = queue.Queue()
    upload_q = queue.Queue() if args.dest_id else None
    
    converter_thread = threading.Thread(
        target=converter_worker, 
        args=(file_q, upload_q),
        daemon=True
    )
    converter_thread.start()
    
    try:
        drive = init_drive()

        upload_thread = None

        if args.dest_id:
            upload_thread = threading.Thread(
                target=uploader_worker,
                args=(drive, args.dest_id, upload_q),
                daemon=True
            )
            upload_thread.start()
        
        print(f"Downloading from Google Drive folder ID: {args.source_id}")
        print(f"Saving to: {args.output_dir}")
        if args.dest_id:
            print(f"Cloud save enabled, uploading to folder ID: {args.dest_id}")
        
        # Start the download process
        download_dir(drive, args.source_id, args.output_dir, file_q, last_date_filter)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up threads
        file_q.put("STOP")
        converter_thread.join()
        
        if args.dest_id:
            upload_q.put("STOP")
            upload_thread.join()
        
        print("\nAll files processed")

if __name__ == "__main__":
    main()