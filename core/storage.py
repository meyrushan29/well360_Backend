
import firebase_admin
from firebase_admin import credentials, storage
import os
import logging
import json
import base64
import uuid

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Handles file storage abstraction.
    If Firebase credentials are set (FIREBASE_ADMIN_SDK_JSON), uploads files to Firebase Storage.
    Otherwise, assumes files are stored locally and returns the local static URL.
    """
    def __init__(self):
        self.bucket = None
        self.bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET") # e.g., your-project.appspot.com

        # Check if Firebase is configured
        firebase_creds_json = os.getenv("FIREBASE_ADMIN_SDK_JSON")
        
        if firebase_creds_json and self.bucket_name:
            try:
                # Parse JSON from env var (it might be a base64 string or raw json)
                try:
                    cred_dict = json.loads(firebase_creds_json)
                except json.JSONDecodeError:
                     # Try decoding base64 if raw json fails (common for multiline secrets)
                     import base64
                     decoded = base64.b64decode(firebase_creds_json).decode('utf-8')
                     cred_dict = json.loads(decoded)

                cred = credentials.Certificate(cred_dict)
                
                # Initialize App (ensure it's not already initialized)
                try:
                    firebase_admin.get_app()
                except ValueError:
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': self.bucket_name
                    })
                
                self.bucket = storage.bucket()
                logger.info("StorageManager: Firebase Storage Initialized.")
            except Exception as e:
                logger.error(f"StorageManager: Failed to initialize Firebase: {e}")
                self.bucket = None
        else:
            logger.info("StorageManager: Running in Local Mode (No Firebase credentials found).")

    def upload_file(self, local_path: str, remote_folder: str, local_url_prefix: str, content_type: str = None) -> str:
        """
        Uploads a file to Firebase Storage if configured, or returns the local static URL.
        
        Args:
            local_path (str): Path to the file on disk (e.g., 'img/uploads/file.png')
            remote_folder (str): Folder in storage bucket (e.g., 'hydration_images')
            local_url_prefix (str): Prefix for local static URL (e.g., '/uploads')
            content_type (str, optional): MIME type. Detected automatically if None.
            
        Returns:
            str: Public signed URL of the file (Firebase URL or local relative URL).
        """
        if not local_path or not os.path.exists(local_path):
            return None

        filename = os.path.basename(local_path)

        # === MODE: FIREBASE STORAGE ===
        if self.bucket:
            # Generate a unique path to avoid collisions if filenames are same
            blob_path = f"{remote_folder}/{filename}"
            
            try:
                blob = self.bucket.blob(blob_path)
                
                # Set content type if provided
                if content_type:
                    blob.content_type = content_type

                blob.upload_from_filename(local_path)
                
                # Make public
                blob.make_public()
                
                # Get public URL
                url = blob.public_url
                
                # Clean up local file since it's now in cloud storage
                try:
                    os.remove(local_path)
                except OSError:
                    pass
                
                return url
            except Exception as e:
                logger.error(f"Firebase Upload Failed: {e}")
                # Fallback to local if upload fails
                return f"{local_url_prefix}/{filename}"

        # === MODE: LOCAL STORAGE ===
        else:
            # File is already locally saved by the calling function, just return the URL
            return f"{local_url_prefix}/{filename}"

# Global instance
storage_manager = StorageManager()
