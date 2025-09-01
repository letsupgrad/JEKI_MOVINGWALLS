import streamlit as st
import json
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
        values.append(round(value, 4))
    return {"indices": indices, "values": values}

def upsert_records(index, tab_name, records):
    vectors = []
    for record in records:
        if record is None:
            continue
        record_id = record.get("Reference ID") or record.get("Reference_Id") or record.get("id")
        if not record_id:
            record_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode('utf-8')).hexdigest()
        sanitized_metadata = {k.lower().replace(' ', '_'): v for k, v in record.items()}
        sparse_vector = generate_sparse_vector_from_id(record_id)
        vectors.append({
            "id": record_id,
            "sparse_values": sparse_vector,
            "metadata": sanitized_metadata
        })
    if vectors:
        st.write(f"Upserting {len(vectors)} vectors for tab '{tab_name}'...")
        index.upsert(vectors=vectors, namespace=tab_name)
        st.success(f"Upsert complete for '{tab_name}'.")
    else:
        st.warning(f"No data to upsert for tab '{tab_name}'.")

def print_tab(tab_name, data):
    st.header(f"Tab: {tab_name}")
    if isinstance(data, list):
        for i, entry in enumerate(data, start=1):
            with st.expander(f"Entry {i}"):
                if isinstance(entry, dict):
                    for key, value in entry.items():
                        st.write(f"**{key}:** {value}")
                else:
                    st.write(entry)
    elif isinstance(data, dict):
        for key, value in data.items():
            st.write(f"**{key}:** {value}")
    else:
        st.write(data)

def main():
    st.title("Pinecone Sparse Vector Upsert & Query")

    api_key = st.text_input("Pinecone API Key", type="password")
    index_name = st.text_input("Pinecone Index Name", value="jeki")
    pinecone_dim = st.number_input("Vector dimension", value=1000, step=1)
    pinecone_region = st.text_input("Pinecone Region", value="us-east-1")

    uploaded_file = st.file_uploader("Upload JSON file", type=["json"])

    if st.button("Initialize Pinecone and Upsert Data") and api_key and index_name and uploaded_file:
        pc = Pinecone(api_key=api_key)
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=pinecone_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=pinecone_region)
            )
            st.info(f"Created index '{index_name}'.")
        index = pc.Index(index_name)

        try:
            json_data = json.load(uploaded_file)
        except Exception as e:
            st.error(f"Error reading JSON: {e}")
            return

        for tab_name, tab_data in json_data.items():
            if tab_data:
                upsert_records(index, tab_name, tab_data)
            else:
                st.warning(f"No data for tab: {tab_name}")

        st.info("Waiting for indexing to complete...")
        time.sleep(5)
        st.success("Upsert completed!")

        st.session_state["index"] = index  # Store index object for later use
        st.session_state["json_data"] = json_data

    if "json_data" in st.session_state:
        st.subheader("Tab Data Preview")
        for tab_name, tab_data in st.session_state["json_data"].items():
            print_tab(tab_name, tab_data)

    st.subheader("Fetch Metadata by Record ID")
    record_id = st.text_input("Enter Record ID to fetch")
    if st.button("Fetch Record") and "index" in st.session_state and record_id:
        index = st.session_state["index"]
        try:
            response = index.fetch(ids=[record_id])
            if response.vectors and record_id in response.vectors:
                metadata = response.vectors[record_id].metadata
                st.write(f"Metadata for Record ID: {record_id}")
                for k, v in metadata.items():
                    st.write(f"**{k}:** {v}")
            else:
                st.warning(f"Record ID {record_id} not found.")
        except Exception as e:
            st.error(f"Error fetching record {record_id}: {e}")

if __name__ == "__main__":
    main()
