from google.cloud import storage
from PIL import Image
from io import BytesIO

from dataclasses import dataclass, field


class GCPDataLoader:

    def __init__(self, bucket_name: str) -> None:
        """"""

        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def download_data(self, data_blob_name: str):
        """"""

        blob = self.bucket.blob(data_blob_name)
        data = blob.download_as_bytes()

        return data


@dataclass
class GCPImageFolder:
    description: list[str] = field(init=False)
    json: dict = field(init=False)
    image: bytes
