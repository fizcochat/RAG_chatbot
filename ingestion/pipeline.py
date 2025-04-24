import os
import json
from pdf_utils import process_pdf
from pinecone_ops import (create_embeddings, upsert_embeddings_to_pinecone,index)
from file_hashing import file_has_changed, update_hash_record, load_hashes, save_hashes, compute_file_hash, get_current_file_hashes
 
"""
# Use if you want to delete all vectors in the index
stats = index.describe_index_stats()
namespaces = stats.get("namespaces", {})
print(namespaces)
if "" in namespaces:  # Default namespace exists
    print("Wiping default namespace...")
    index.delete(delete_all=True, namespace="")
else:
    print("No vectors found in default namespace. Skipping delete.")
print("Index Deletec")
"""

CHUNK_ID_TRACKER = "indexed_chunks.json"

def load_chunk_map():
    if os.path.exists(CHUNK_ID_TRACKER):
        with open(CHUNK_ID_TRACKER, "r") as f:
            return json.load(f)
    return {}

def save_chunk_map(chunk_map):
    with open(CHUNK_ID_TRACKER, "w") as f:
        json.dump(chunk_map, f, indent=2)

def run_ingestion_pipeline():
    pdf_dir = "../data_documents"
    file_paths = sorted([os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
    existing_hashes = load_hashes()
    current_hashes = get_current_file_hashes()
    chunk_map = load_chunk_map()

    for file_path in file_paths:
        file_hash = compute_file_hash(file_path)
        if existing_hashes.get(file_path) != file_hash:
            print(f"🔁 Change detected in {file_path}, reprocessing...")
            texts = process_pdf(file_path)
            embeddings = create_embeddings(texts)
            ids = [f"{file_hash}_chunk_{i}" for i in range(len(embeddings))]

            # Delete old IDs by hash if available
            old_ids = chunk_map.get(file_hash, [])
            if old_ids:
                index.delete(ids=old_ids)

            upsert_embeddings_to_pinecone(index, embeddings, ids, texts, file_hash)
            chunk_map[file_hash] = ids
            existing_hashes[file_path] = file_hash
            update_hash_record(file_path)
        else:
            print(f"✅ {file_path} unchanged, skipping.")

    # Detect deleted files and remove their vectors from Pinecone
    to_delete = set(existing_hashes.keys()) - set(current_hashes.keys())
    for deleted_path in to_delete:
        deleted_hash = existing_hashes[deleted_path]
        print(f"🗑️ File removed locally, deleting from Pinecone: {deleted_path}")
        if deleted_hash in chunk_map:
            index.delete(ids=chunk_map[deleted_hash])
            del chunk_map[deleted_hash]
        del existing_hashes[deleted_path]

    save_hashes(existing_hashes)
    save_chunk_map(chunk_map)