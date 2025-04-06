# Usage

0. If step 0 then other steps are not necessary: ask creator for his api keys.

1. Go to APIs Console and make your own project.

2. Search for ‘Google Drive API’, select the entry, and click ‘Enable’.

3. Select ‘Credentials’ from the left menu, click ‘Create Credentials’, select ‘OAuth client ID’.

4. Now, the product name and consent screen need to be set -> click ‘Configure consent screen’ and follow the instructions. Once finished:

5. Select ‘Application type’ to be Native application.

6. Enter an appropriate name.

7. Input http://localhost:8080/ for ‘Authorized redirect URIs’.

8. Click ‘Create’.

9. Click ‘Download JSON’ on the right side of Client ID to download client_secret_<really long ID>.json.

10. The downloaded file has all authentication information of your application. Rename the file to “client_secrets.json” and place it in your working directory.

## Script options

| Argument      | Required | Description                                                                 |
|---------------|----------|-----------------------------------------------------------------------------|
| `--source-id` | Yes      | Google Drive folder ID to download from                                     |
| `--output-dir`| No       | Local output directory (default: 'dataset')                                 |
| `--last-date` | No       | Only process files created after this date (format: DD-MM-YYYY)             |
| `--dest-id`   | No       | Google Drive folder ID for cloud backup (enables upload automatically)      |
