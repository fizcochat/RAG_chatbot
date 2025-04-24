import os
import hashlib
import json

HASH_FILE = "document_hashes.json"

def compute_file_hash(file_path: str) -> str:
    """Compute the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_hashes() -> dict:
    """Load the saved hashes from JSON file."""
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}


def save_hashes(hashes: dict):
    """Save the updated hashes to JSON."""
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def file_has_changed(file_path: str) -> bool:
    """Check if a file has changed since the last run."""
    current_hash = compute_file_hash(file_path)
    saved_hashes = load_hashes()
    old_hash = saved_hashes.get(file_path)
    return current_hash != old_hash


def update_hash_record(file_path: str):
    """Update the hash record after reprocessing a file."""
    current_hash = compute_file_hash(file_path)
    saved_hashes = load_hashes()
    saved_hashes[file_path] = current_hash
    save_hashes(saved_hashes)


def get_current_file_hashes():
    current_hashes = {}
    for root, _, files in os.walk("../data_documents"):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                current_hashes[path] = compute_file_hash(path)
    return current_hashes
