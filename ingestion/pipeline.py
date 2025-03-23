import os
from pdf_utils import process_pdf
from pinecone_ops import create_embeddings, upsert_embeddings_to_pinecone, delete_existing_chunks, index
from file_hashing import file_has_changed, update_hash_record

CHUNK_ID_TRACKER = "indexed_chunks.json"

def load_chunk_map():
    if os.path.exists(CHUNK_ID_TRACKER):
        import json
        with open(CHUNK_ID_TRACKER, "r") as f:
            return json.load(f)
    return {}

def save_chunk_map(chunk_map):
    import json
    with open(CHUNK_ID_TRACKER, "w") as f:
        json.dump(chunk_map, f, indent=2)

def run_ingestion_pipeline():
    pdf_dir = "../data_documents"
    file_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    chunk_map = load_chunk_map()

    for file_path in file_paths:
        if file_has_changed(file_path):
            print(f"🔁 Change detected in {file_path}, reprocessing...")
            texts = process_pdf(file_path)
            embeddings = create_embeddings(texts)
            ids = [f"{file_path}_chunk_{i}" for i in range(len(embeddings))]

            # Delete old chunks
            old_ids = chunk_map.get(file_path, [])
            if old_ids:
                delete_existing_chunks(index, old_ids)

            # Upload new ones
            upsert_embeddings_to_pinecone(index, embeddings, ids, texts)
            chunk_map[file_path] = ids
            update_hash_record(file_path)
        else:
            print(f"✅ {file_path} unchanged, skipping.")

    save_chunk_map(chunk_map)

