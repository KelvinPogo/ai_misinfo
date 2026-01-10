from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize and authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# Download the Deepfake Image Detection dataset into current folder
api.dataset_download_files(
    'saurabhbagchi/deepfake-image-detection',
    path='.',
    unzip=True
)

print("Download complete!")