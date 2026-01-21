import os
import hashlib
import shutil
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

@runtime_checkable
class PDFStorageProvider(Protocol):
    """Protocol defining how raw PDF files should be stored and retrieved."""
    def upload_file(self, local_path: str, remote_name: str) -> str: ...
    def download_file(self, remote_name: str, local_destination: str) -> str: ...
    def list_files(self) -> List[str]: ...

class PDFStorage:
    """Implementation for storing PDFs on the local file system (Dev/Test)."""
    def __init__(self, base_dir: str) -> None:
        self.base_dir: Path = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(self, local_path: str, remote_name: str) -> str:
        """Copies the local file to the storage directory with the given name."""
        destination: Path = self.base_dir / remote_name
        # Using shutil.copy2 to preserve metadata and improve efficiency
        shutil.copy2(local_path, destination)
        return str(destination)

    def download_file(self, remote_name: str, local_destination: str) -> str:
        """Retrieves the file from storage to a local path."""
        source: Path = self.base_dir / remote_name
        shutil.copy2(source, local_destination)
        return local_destination

    def list_files(self) -> List[str]:
        """Returns filenames of all PDFs in the storage directory."""
        return [f.name for f in self.base_dir.glob("*.pdf")]

class PDFStoreManager:
    def __init__(self, storage_provider: PDFStorageProvider) -> None:
        """
        Manages the persistence of raw PDF documents using content-based hashing.
        """
        self.storage: PDFStorageProvider = storage_provider

    def _get_file_hash(self, file_path: str) -> str:
        """
        Generates a SHA-256 hash of the file content for naming.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def save_pdf(self, file_path: str, existing_hash: Optional[str] = None) -> str:
            """
            Saves a PDF. Uses existing_hash if provided, otherwise calculates it.
            """
            content_hash: str = existing_hash if existing_hash else self._get_file_hash(file_path)
            remote_name: str = f"{content_hash}.pdf"
            return self.storage.upload_file(file_path, remote_name)

    def get_pdf_for_processing(self, remote_name: str, temp_download_path: str = "/tmp") -> str:
        """
        Downloads a PDF from storage to a temporary local path.
        """
        local_target: str = os.path.join(temp_download_path, remote_name)
        return self.storage.download_file(remote_name, local_target)

    def list_all_pdfs(self) -> List[str]:
        """Returns a list of all stored PDF names (hashes)."""
        return self.storage.list_files()