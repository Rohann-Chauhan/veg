import zipfile

import os


# Current data folder
base_path = os.path.dirname(
    __file__
)


# ZIP path
zip_path = os.path.join(
    base_path,

    "archive(5).zip"
)


# Extraction folder
extract_path = os.path.join(

    base_path,

    "cloud_dataset"

)


# Create folder
os.makedirs(

    extract_path,

    exist_ok=True

)


# Extract
with zipfile.ZipFile(

    zip_path,

    "r"

) as zip_ref:


    zip_ref.extractall(

        extract_path

    )


print(
    "Unzipped successfully!"
)