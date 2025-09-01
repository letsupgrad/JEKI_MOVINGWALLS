import streamlit as st
import json
import os
import hashlib
import time
from pinecone import Pinecone, ServerlessSpec

def generate_sparse_vector_from_id(record_id: str, num_indices=5, max_index=1000):
    h = hashlib.sha256(record_id.encode('utf-8')).hexdigest()
    seen_indices = set()
    for i in range(num_indices * 4):
        if len(seen_indices) >= num_indices:
            break
        index = int(h[i*2: i*2+4], 16) % max_index
        seen_indices.add(index)
    indices = sorted(list(seen_indices))
    values = []
    for i in range(len(indices)):
        value = 0.1 + (int(h[i*4: i*4+4], 16) % 900) / 1000.0
        values.append(round(value,4))
    return {"indices": indices, "values": values}

def upsert_records(index, tab_name, records):
    vectors = []
    for record in records:
        if record is None:
            continue
        record_id = record.get("Reference ID") or record.get("Reference_Id") or record.get("id")
        if not record_id:
            record_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode('utf-8')).hexdigest()
        sanitized_metadata = {k.lower().replace(' ', '_'): v for k,v in record.items()}
        sparse_vector = generate_sparse_vector_from_id(record_id)
        vectors.append({
            "id": record_id,
            "sparse_values": sparse_vector,
            "metadata": sanitized_metadata
        })
    if vectors:
        index.upsert(vectors=vectors)
        st.success(f"Upserted {len(vectors)} vectors for tab '{tab_name}'")
    else:
        st.warning(f"No data to upsert for tab '{tab_name}'")

def print_tab(title, data):
    st.header(title)
    if isinstance(data, list):
        for i, entry in enumerate(data, 1):
            with st.expander(f"Entry {i}"):
                if isinstance(entry, dict):
                    for key,val in entry.items():
                        st.write(f"**{key}**: {val}")
                else:
                    st.write(entry)
    elif isinstance(data, dict):
        for k,v in data.items():
            st.write(f"**{k}**: {v}")
    else:
        st.write(data)

st.title("Pinecone JSON Viewer & Upserter")

JSON_DIR = "./json_files"  # Change as per your server

filename = st.text_input("Enter JSON filename to load (must be in server directory):")

if filename:
    filepath = os.path.join(JSON_DIR, filename)
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            st.success(f"Loaded '{filename}' successfully!")

            # Display JSON content separated by tabs
            for tab_name, tab_data in json_data.items():
                print_tab(tab_name, tab_data)

            # Pinecone Upsert Section
            with st.expander("Upsert JSON data into Pinecone"):
                api_key = st.text_input("Pinecone API Key", type="password", key="pinecone_key")
                index_name = st.text_input("Pinecone Index Name", value="jeki", key="pinecone_index")

                if st.button("Upsert Data to Pinecone"):
                    if not api_key:
                        st.error("Pinecone API key is required!")
                    else:
                        pc = Pinecone(api_key=api_key)
                        if not pc.has_index(index_name):
                            pc.create_index(
                                name=index_name,
                                dimension=1000,
                                metric="cosine",
                                spec=ServerlessSpec(cloud="aws", region="us-east-1")
                            )
                            st.info(f"Created Pinecone index '{index_name}'.")
                        index = pc.Index(index_name)

                        for tab_name, tab_data in json_data.items():
                            if tab_data:
                                upsert_records(index, tab_name, tab_data)
                            else:
                                st.warning(f"No data in tab '{tab_name}' for upsert")

                        st.success("Upsert completed! Waiting index stabilization...")
                        time.sleep(5)

                        # Show sample data fetched from Pinecone for verification
                        sample_id = st.text_input("Enter Sample Record ID to fetch from Pinecone (optional)", key="sample_id")
                        if sample_id:
                            try:
                                response = index.fetch(ids=[sample_id])
                                if response.vectors and sample_id in response.vectors:
                                    metadata = response.vectors[sample_id].metadata
                                    print_tab(f"Metadata for {sample_id}", metadata)
                                else:
                                    st.warning(f"Record ID '{sample_id}' not found in Pinecone.")
                            except Exception as e:
                                st.error(f"Error fetching from Pinecone: {e}")

        except Exception as e:
            st.error(f"Error loading JSON file: {e}")
    else:
        st.error(f"File '{filename}' not found in directory '{JSON_DIR}'.")
else:
    st.info("Please enter a JSON filename to load.")

