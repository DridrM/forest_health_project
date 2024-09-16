import os

#################################
# Params for the dataset module #
#################################

GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")

GCP_BUCKET_NAME = os.environ.get("GCP_BUCKET_NAME")

# Default blending ratio to blend the mask with the rgb image
DEFAULT_RGB_BLENDING_RATIO = 0.1
