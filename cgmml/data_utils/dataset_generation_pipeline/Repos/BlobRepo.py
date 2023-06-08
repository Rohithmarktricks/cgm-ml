import datetime
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient


class BlobRepo:
    def download_from_blob_storage(src: str, dest: str, container: str, blob_client: BlobServiceClient) -> None:
        """
        Download a file from Azure Blob Storage and save it to the specified destination.

        Args:
        src (str): The source path of the file in the Blob Storage.
        dest (str): The destination path where the downloaded file will be saved.
        container (str): The name of the container in Blob Storage.
        blob_client (BlobServiceClient): An instance of the Azure BlobServiceClient.

        Returns:
        None
        """
        # Get the BlobClient for the specified container and blob (file)
        blob = blob_client.get_blob_client(container=container, blob=src)

        # Create the destination directory if it doesn't exist
        Path(dest).parent.mkdir(parents=True, exist_ok=True)

        # Download the file from Blob Storage and save it to the destination
        with open(dest, "wb") as download_file:
            content = blob.download_blob().readall()
            download_file.write(content)

    def upload_to_blob_storage(connection_str_des: str, processed_fnames, dataset_type: str, data_category: str):
        """
        This function uploads given files to an Azure Blob storage container.

        The function uses a multi-threaded approach to upload the files concurrently, which
        can significantly improve the upload speed for large datasets. The data is organized
        in the Blob storage based on its type and category and is stored under a unique directory
        name that is derived from the current timestamp.

        Parameters:
        connection_str_des (str): The connection string of the destination Blob storage.
        processed_fnames (list): A list of file names (with absolute paths) that are to be uploaded.
        dataset_type (str): The type of the dataset (used for organizing the data in Blob storage).
        data_category (str): The category of the data (used for organizing the data in Blob storage).

        Returns:
        list: A list of results from ThreadPool's map function which contains None for successful uploads.
            Any exceptions raised during the upload will be present in this list.
        """
        CONTAINER_NAME_DEST_SA = "cgm-datasets"
        BLOB_SERVICE_CLIENT_DSET = BlobServiceClient.from_connection_string(connection_str_des)
        DATASET_NAME = "dataset"
        dest_dir = datetime.now(datetime.timezone.utc) \
            .strftime(f"{DATASET_NAME}-{dataset_type}-{data_category}-%Y-%m-%d-%H-%M-%S")
        PREFIX = "/dbfs/tmp/env_prod/"

        def remove_prefix(text: str, prefix: str) -> str:
            if text.startswith(prefix):
                return text[len(prefix):]
            return text

        def upload_to_blob_storage_intern(src: str, dest_container: str, dest_fpath: str):
            blob_client = BLOB_SERVICE_CLIENT_DSET.get_blob_client(container=dest_container, blob=dest_fpath)
            with open(src, "rb") as data:
                blob_client.upload_blob(data, overwrite=False)

        def _upload(full_name):
            assert PREFIX in full_name, full_name
            dest_fpath = os.path.join(dest_dir, remove_prefix(full_name, PREFIX))
            upload_to_blob_storage_intern(src=full_name, dest_container=CONTAINER_NAME_DEST_SA, dest_fpath=dest_fpath)

        NUM_THREADS = 64
        pool = ThreadPool(NUM_THREADS)
        results = pool.map(_upload, processed_fnames)

        return results
