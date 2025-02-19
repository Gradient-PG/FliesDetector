import os
import io
from queue import Queue
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from PIL import Image

try:
    import pyheif
except ImportError:
    pyheif = None

# Authenticate with OAuth2 with Google
gauth = GoogleAuth()
gauth.CommandLineAuth()

# Connect to Google Drive account
drive = GoogleDrive(gauth)

# Needed variables
source_folder_id = '1tXWu9ic3p2J_NIHmg0ugnPtB8VFvvpjC'
dest_folder_name = 'label_studio_ready'
parent_dest_folder = '1c6ldihcjxO8ppgNRTfZxYqlzoKuFmXGN'
VALID_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'tiff', 'tif', 'heic', 'heif', 'bmp'}
FOLDER_MIMETYPE = 'application/vnd.google-apps.folder'

def remove_duplicates_from_folder(folder_id):
    md5_dict = {}
    subfolder_list = []

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    for file in file_list:
        if file['mimeType'] == FOLDER_MIMETYPE:
            subfolder_list.append(file['id'])
            continue

        file_md5 = file.get('md5Checksum')

        if not file_md5:
            print(f"Skipping file without MD5 checksum: {file['title']}")
            continue

        if file_md5 in md5_dict:
            print(f"Duplicate found: {file['title']} (ID: {file['id']})")
            file.Delete()
            print(f"Deleted duplicate: {file['title']} (ID: {file['id']})")

        else:
            md5_dict[file_md5] = file['id']

    for subfolder_id in subfolder_list:
        remove_duplicates_from_folder(subfolder_id)


def convert_to_jpg(file_content, file_extension):
    try:
        # Handle HEIC
        if file_extension.lower() in ['heic', 'heif'] and pyheif:
            heif_file = pyheif.read(file_content)
            img = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        
        else:
            img = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Save to bytes buffer
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95, optimize=True)
        output.seek(0)
        return output
    
    except Exception as e:
        print(f"Error converting file: {str(e)}")
        return None

def create_folder(drive, folder_name, parent_id=None):
    # Create folder in Google Drive
    folder = drive.CreateFile({
        'title': folder_name,
        'mimeType': FOLDER_MIMETYPE,
        'parents': [{'id': parent_id} if parent_id else []]
    })

    # Upload created folder
    folder.Upload()

    # Return uploaded folder ID which will be used as a handle
    return folder['id'] 

def process_folder(source_id, dest_id):
    # Recursive folder processing
    q = Queue()
    q.put((source_id, dest_id))

    while not q.empty():
        curr_src, curr_dest = q.get()

        file_list = drive.ListFile({
            'q': f"'{curr_src}' in parents and trashed=false"
        }).GetList()

        for item in file_list:
            if item['mimeType'] == FOLDER_MIMETYPE:
                # If we found folder then we have to create corresponding one in destination folder
                # And put folder in queue to process
                new_folder_id = create_folder(drive, item['title'], curr_dest)
                q.put((item['id'], new_folder_id))

                print(f"Created folder: {item['title']}")

            elif item['mimeType'].startswith('application/vnd.google-apps'):
                # Avoid processing Google app docs, bcs they make errors
                print(f"Skipping Google Document: {file['title']}")
            
            else:
                # If we found file we have to process it accordingly to format
                # And paste processed file in destination location
                file_extension = item['title'].split('.')[-1].lower()

                if file_extension in VALID_EXTENSIONS:
                    print(f"Processing: {item['title']}")

                    try:
                        # Download file to local folder
                        file = drive.CreateFile({'id': item['id']})
                        file.FetchContent()
                        content = file.content.read()

                        converted = convert_to_jpg(content, file_extension)

                        if converted:
                            # Create new filename
                            base_name = os.path.splitext(item['title'])[0]
                            new_filename = f"{base_name}.jpg"

                            # Upload converted file to destination
                            new_file = drive.CreateFile({
                                'title': new_filename,
                                'parents': [{'id': curr_dest}],
                                'mimeType': 'image/jpeg'
                            })

                            new_file.content = converted
                            new_file.Upload()
                            print(f"Converted and uploaded: {new_filename}")

                    except Exception as e:
                        print(f"Error processing: {item['title']}: {str(e)}")

def main():
    # Create root destinanation folder
    dest_folder_id = create_folder(drive, dest_folder_name, parent_dest_folder)
    print(f"Created destination folder: {dest_folder_name}")

    # Process source folder 
    process_folder(source_folder_id, dest_folder_id)

    # At the end remove duplicates on folder level
    # We have to do it afterwards, because there was some error with permissions for some files
    remove_duplicates_from_folder(dest_folder_id)

if __name__ == '__main__':
    main()



