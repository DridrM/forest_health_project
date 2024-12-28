import os

#################################
# Params for the dataset module #
#################################

GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")

GCP_BUCKET_NAME = os.environ.get("GCP_BUCKET_NAME")

LOCAL_ROOT_DATA_PATH = os.environ.get("LOCAL_ROOT_DATA_PATH", "/")

# Default blending ratio to blend the mask with the rgb image
DEFAULT_RGB_BLENDING_RATIO = 0.1

# Dict that indicate if the rgb image should be blended with the mask given the data status
BLEND_GIVEN_DATA_STATUS = {"train": True, "valid": False, "test": False}

# Paths of the metadata csv files inside the GCP bucket
GCP_TRAIN_METADATA_PATH = "train.csv"
GCP_VALID_METADATA_PATH = "val.csv"
GCP_TEST_METADATA_PATH = "test.csv"

# Dict used to create the new data_status column inside the local "metadata" csv file
GCP_METADATA_PATH_AND_LABELS = {
    "train": GCP_TRAIN_METADATA_PATH,
    "valid": GCP_VALID_METADATA_PATH,
    "test": GCP_TEST_METADATA_PATH,
}

# Paths of rgb image and mask inside a GCP image folder
GCP_RGB_IMAGE_PATH = "images/visible/composite.png"
GCP_MASK_PATH = "forest_loss_region.pkl"

# Local csv metadata path
LOCAL_METADATA_PATH = f"{LOCAL_ROOT_DATA_PATH}/metadata.csv"

# Local data folders paths
LOCAL_EXAMPLES_PATH = f"{LOCAL_ROOT_DATA_PATH}/examples"
LOCAL_TRAIN_FOLDER = f"{LOCAL_EXAMPLES_PATH}/train"
LOCAL_VALID_FOLDER = f"{LOCAL_EXAMPLES_PATH}/valid"
LOCAL_TEST_FOLDER = f"{LOCAL_EXAMPLES_PATH}/test"

# Dict containing the local data path
LOCAL_DATA_FOLDERS = {
    "train": LOCAL_TRAIN_FOLDER,
    "valid": LOCAL_VALID_FOLDER,
    "test": LOCAL_TEST_FOLDER,
}

# Names of the local rgb images and mask images
LOCAL_RGB_IMAGE_NAME = "rgb_image.png"
LOCAL_MASK_IMAGE_NAME = "mask_image.png"


###############################
# Params for the train module #
###############################

LOCAL_PROJECT_PATH = os.environ.get("LOCAL_PROJECT_PATH")

TRAINED_MODELS_FOLDER = f"{LOCAL_PROJECT_PATH}/trained_models"
