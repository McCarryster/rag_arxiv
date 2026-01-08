import os
# import boto3
from typing import List, Protocol, Optional, runtime_checkable
from pathlib import Path
import config

@runtime_checkable
class PDFStorageProvider(Protocol):
    """Protocol defining how raw PDF files should be stored and retrieved."""
    def upload_file(self, local_path: str, remote_name: str) -> str: ...
    def download_file(self, remote_name: str, local_destination: str) -> str: ...
    def list_files(self) -> List[str]: ...

class LocalPDFStorage:
    """Implementation for storing PDFs on the local file system (Dev/Test)."""
    def __init__(self, base_dir: str) -> None:
        self.base_dir: Path = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(self, local_path: str, remote_name: str) -> str:
        destination: Path = self.base_dir / remote_name
        # Simple copy logic
        with open(local_path, "rb") as src, open(destination, "wb") as dst:
            dst.write(src.read())
        return str(destination)

    def download_file(self, remote_name: str, local_destination: str) -> str:
        source: Path = self.base_dir / remote_name
        with open(source, "rb") as src, open(local_destination, "wb") as dst:
            dst.write(src.read())
        return local_destination

    def list_files(self) -> List[str]:
        return [f.name for f in self.base_dir.glob("*.pdf")]

# class S3PDFStorage:
#     """Implementation for storing PDFs in AWS S3 (Production)."""
#     def __init__(self, bucket_name: str, prefix: str = "uploads/") -> None:
#         self.s3 = boto3.client('s3')
#         self.bucket: str = bucket_name
#         self.prefix: str = prefix

#     def upload_file(self, local_path: str, remote_name: str) -> str:
#         s3_key: str = f"{self.prefix}{remote_name}"
#         self.s3.upload_file(local_path, self.bucket, s3_key)
#         return f"s3://{self.bucket}/{s3_key}"

#     def download_file(self, remote_name: str, local_destination: str) -> str:
#         s3_key: str = f"{self.prefix}{remote_name}"
#         self.s3.download_file(self.bucket, s3_key, local_destination)
#         return local_destination

#     def list_files(self) -> List[str]:
#         response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)
#         if 'Contents' not in response:
#             return []
#         return [obj['Key'].replace(self.prefix, "") for obj in response['Contents']]

class PDFStoreManager:
    def __init__(self, storage_provider: PDFStorageProvider) -> None:
        """
        Manages the persistence of raw PDF documents.
        
        Args:
            storage_provider: An implementation of PDFStorageProvider (Local or S3).
        """
        self.storage: PDFStorageProvider = storage_provider

    def save_pdf(self, file_path: str) -> str:
        """
        Saves a PDF to the configured storage.
        
        Args:
            file_path: The path to the file on the local machine/container.
            
        Returns:
            str: The final URI or path of the stored file.
        """
        file_name: str = os.path.basename(file_path)
        return self.storage.upload_file(file_path, file_name)

    def get_pdf_for_processing(self, remote_name: str, temp_download_path: str = "/tmp") -> str:
        """
        Downloads a PDF to a temporary local path so that loaders (like PyPDFLoader) 
        can read it.
        """
        local_target: str = os.path.join(temp_download_path, remote_name)
        return self.storage.download_file(remote_name, local_target)

    def list_all_pdfs(self) -> List[str]:
        """Returns a list of all stored PDF names."""
        return self.storage.list_files()


def get_pdf_store_manager() -> PDFStoreManager:
    if not config.PROD:
        storage_provider = LocalPDFStorage(base_dir="/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/vector_db")
    else:
        ...

    return PDFStoreManager(storage_provider=storage_provider)