from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload,MediaFileUpload
from googleapiclient.discovery import build
import pprint as pp
import io
import shutil


def createRemoteFolder(self, folderName, parentID = None):
        # Create a folder on Drive, returns the newely created folders ID
    body = {
          'name': folderName,
          'mimeType': "application/vnd.google-apps.folder"
    }
    if parentID:
        body['parents'] = [parentID]
    root_folder = drive_service.files().create(body = body).execute()
    return root_folder['id']


SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = '/home/axeka/VSCodeProjects/NLP_Emotions/NLP_Emotions/gdrive_secrets.json'

credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

query = f"name='colabdrive' and mimeType='application/vnd.google-apps.folder'"
results = service.files().list(
    pageSize=10, 
    fields="nextPageToken, files(id, name, mimeType, parents, createdTime)",
    q=query).execute()
if not results['files']:
    folder_id = createRemoteFolder('colabdrive')
else:
    folder_id = '1gF6NfN_cdVPEwkkh3dh06FsRHavwIGlk'
    print(folder_id)

zip_name = 'NLP.zip'
shutil.make_archive('NLP', 'zip', '/home/axeka/VSCodeProjects/NLP_Emotions/NLP_Emotions')
file_path = '/home/axeka/VSCodeProjects/NLP_Emotions/NLP_Emotions/NLP.zip'
file_metadata = {
                'name': zip_name,
                'parents': [folder_id]
            }
query = f"name='NLP.zip' and mimeType='application/zip'"
results = service.files().list(pageSize=10,
                               fields="nextPageToken, files(id, name, mimeType)").execute()
results = service.files().list(
    pageSize=10, 
    fields="nextPageToken, files(id, name, mimeType, parents, createdTime)",
    q=query).execute()
if results['files']:
    media = MediaFileUpload(file_path, resumable=True)
    r = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print('Uploaded NLP.zip')
#else:
#print('File already exists','id:' + results['files'][0]['id'])



