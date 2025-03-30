#!/usr/bin/env python3

import threading
import argparse
import queue
import piexif
import os
import sys
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from PIL import Image
import pillow_heif 

pillow_heif.register_heif_opener()

# Constants
PHOTO_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.heic', '.heif', '.bmp', '.tiff', '.webp'}
DIR_MIME = 'application/vnd.google-apps.folder'

def parse_args():
    parser = argparse.ArgumentParser(description='Download and convert Google Drive photos')
    parser.add_argument('--source-id', required=True, type=str, help='Google Drive source folder ID')
    parser.add_argument('--output-dir', default='dataset', type=str, help='Output directory (default: dataset)')
    parser.add_argument('--last-date', type=str, help='Only process files created after this date (DD-MM-YYYY)')
    return parser.parse_args()

def init_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

def convert_to_jpg(file_path):
    try:
        with Image.open(file_path) as img:
            img_no_exif = Image.new(img.mode, img.size)
            img_no_exif.putdata(list(img.getdata()))

            new_path = os.path.splitext(file_path)[0] + ".jpg"
            img_no_exif.save(new_path, "JPEG", quality=95, subsampling=0)

        print(f"Converted: {os.path.basename(file_path)} -> {os.path.basename(new_path)}")
        os.remove(file_path)
    except Exception as e:
        print(f"Error converting {os.path.basename(file_path)}: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)

def converter_worker(file_q):
    while True:
        file_path = file_q.get()
        if file_path == "STOP":
            break
        convert_to_jpg(file_path)
        file_q.task_done()

def parse_date(date_str):
    return datetime.strptime(date_str, "%d-%m-%Y")

def get_date_dir_name(item):
    created_date = item['createdDate']
    dt = datetime.strptime(created_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    return dt.strftime("%d-%m-%Y")

def should_process_file(item, last_date_filter=None):
    if not last_date_filter:
        return True
    
    created_date = datetime.strptime(item['createdDate'], "%Y-%m-%dT%H:%M:%S.%fZ")
    return created_date > last_date_filter

def download_dir(drive, source_dir_id, local_dir, file_q, last_date_filter=None):
    file_list = drive.ListFile({'q': f"'{source_dir_id}' in parents and trashed=false"}).GetList()
    
    for item in file_list:
        if item['mimeType'] == DIR_MIME:
            download_dir(drive, item['id'], local_dir, file_q, last_date_filter)
        else:
            file_ext = os.path.splitext(item['title'])[1].lower()
            if file_ext in PHOTO_EXTENSIONS and should_process_file(item, last_date_filter):
                date_dir = get_date_dir_name(item)
                target_dir = os.path.join(local_dir, date_dir)
                os.makedirs(target_dir, exist_ok=True)
                
                target_path = os.path.join(target_dir, item['title'])
                if not os.path.exists(target_path):
                    print(f"Downloading: {item['title']}")
                    try:
                        item.GetContentFile(target_path)
                        file_q.put(target_path)
                    except Exception as e:
                        print(f"Error downloading {item['title']}: {str(e)}")

def main():
    args = parse_args()
    
    last_date_filter = None
    if args.last_date:
        try:
            last_date_filter = parse_date(args.last_date)
            print(f"Processing files created after: {last_date_filter.strftime('%d-%m-%Y')}")
        except ValueError:
            print("Invalid date format. Please use DD-MM-YYYY", file=sys.stderr)
            sys.exit(1)

    file_q = queue.Queue()
    converter_thread = threading.Thread(target=converter_worker, args=(file_q,), daemon=True)
    converter_thread.start()

    try:
        drive = init_drive()
        print(f"Downloading from Google Drive folder ID: {args.source_id}")
        print(f"Saving to: {args.output_dir}")
        download_dir(drive, args.source_id, args.output_dir, file_q, last_date_filter)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        file_q.put("STOP")
        converter_thread.join()
        print("\nAll files processed")

if __name__ == "__main__":
    main()