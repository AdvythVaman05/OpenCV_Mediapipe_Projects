import requests
import zipfile
import os

# Define the URL and the local filename
url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
local_zip_file = "vosk-model-small-en-us-0.15.zip"
extraction_path = "vosk-model-small-en-us-0.15"

# Download the file
print("Downloading Vosk model...")
response = requests.get(url)
with open(local_zip_file, 'wb') as file:
    file.write(response.content)
print("Download complete.")

# Unzip the file
print("Unzipping the model...")
with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)
print("Model unzipped successfully.")
