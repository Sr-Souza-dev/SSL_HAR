import urllib.request
import zipfile
import os

def _fetch_and_unzip_dataset(root_data_dir, url, zip_name = None) -> None:
    if not os.path.exists(root_data_dir):
        print(f"Creating the root data directory: [{root_data_dir}]")
        os.makedirs(root_data_dir)

    if zip_name:
        zip_file = f"{root_data_dir}/{zip_name}"
        if zip_name and not os.path.exists(zip_file):

            print(f"Could not find the zip file [{zip_file}]")
            print(f"Trying to download it.")
            urllib.request.urlretrieve(url, zip_file)

        # extract data
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(root_data_dir)
        print("Data downloaded and extracted")

    else:
        urllib.request.urlretrieve(url)

def fetch_data(url, root_data_dir, files, type, zip_name = None) -> None:
    if not root_data_dir or not type or not url:
        print("The parameters <root_data_dir, type, url> are required")
        return
    
    for file in files:
        path = f"{root_data_dir}/{file}.{type}"
        if not os.path.exists(path):
            print("One or more files are missing!")
            _fetch_and_unzip_dataset(root_data_dir, url, zip_name)