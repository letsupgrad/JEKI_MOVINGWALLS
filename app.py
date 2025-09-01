

import streamlit as st
import json
import hashlib
import pandas as pd
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
            # Generate ID from full record hash if no ID field
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
        index.upsert(vectors=vectors)
        st.write(f"Upsert complete for '{tab_name}'.")
    else:
        st.write(f"No data to upsert for tab '{tab_name}'.")

def print_tab(title, data):
    st.header(title)
    if isinstance(data, list):
        # Detect if data is list of dicts (tabular)
        if data and all(isinstance(item, dict) for item in data):
            # Display as a dataframe
            df = pd.DataFrame(data)
            st.dataframe(df)
            # Create simple bar chart for numeric columns if any
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                st.subheader("Charts")
                for col in numeric_cols:
                    st.bar_chart(df[col])
        else:
            for i, entry in enumerate(data, start=1):
                with st.expander(f"Entry {i}"):
                    if isinstance(entry, dict):
                        for key, value in entry.items():
                            st.write(f"**{key}**: {value}")
                    else:
                        st.write(entry)
    elif isinstance(data, dict):
        for key, value in data.items():
            st.write(f"**{key}**: {value}")
    else:
        st.write(data)

def query_data(json_data, query):
    """Simple query: filter records containing query substring in any string metadata."""
    results = {}
    query_lower = query.lower()
    for tab_name, tab_data in json_data.items():
        if isinstance(tab_data, list):
            filtered = []
            for record in tab_data:
                # Check if any string field contains query substring
                if any(query_lower in str(v).lower() for v in record.values()):
                    filtered.append(record)
            if filtered:
                results[tab_name] = filtered
        elif isinstance(tab_data, dict):
            # For dict data, check if query in any value
            if any(query_lower in str(v).lower() for v in tab_data.values()):
                results[tab_name] = tab_data
    return results

st.title("Upload up to 10 JSON files, view, analyze, and upsert to Pinecone")

uploaded_files = st.file_uploader(
    "Upload JSON files", type="json", accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("Please upload no more than 10 files.")
    else:
        filenames = [file.name for file in uploaded_files]
        selected_files = st.multiselect("Select file(s) to view and analyze", options=filenames)
        file_map = {file.name: file for file in uploaded_files}

        json_data_all = {}
        for fname in selected_files:
            file = file_map.get(fname)
            if file:
                try:
                    file.seek(0)
                    json_data = json.load(file)
                    json_data_all[fname] = json_data
                    st.subheader(f"Contents of {fname}")
                    for tab_name, tab_data in json_data.items():
                        print_tab(tab_name, tab_data)
                except Exception as e:
                    st.error(f"Error reading {fname}: {e}")

        # Query interface on combined JSON data from selected files
        if selected_files:
            st.subheader("Query data across selected files")
            query_text = st.text_input("Enter search query (case insensitive substring match)")
            if query_text:
                combined_results = {}
                for data in json_data_all.values():
                    res = query_data(data, query_text)
                    combined_results.update(res)
                if combined_results:
                    st.write(f"Query results for '{query_text}':")
                    for tab_name, results in combined_results.items():
                        print_tab(tab_name, results)
                else:
                    st.write("No matching records found.")

        # Pinecone upsert section
        if st.checkbox("Upsert selected data to Pinecone"):
            api_key = st.text_input("Enter Pinecone API Key:", type="password")
            index_name = st.text_input("Enter Pinecone Index Name:", value="jeki")

            if st.button("Run Upsert"):
                if not (api_key and selected_files):
                    st.error("Please select files and enter API key.")
                else:
                    pc = Pinecone(api_key=api_key)
                    if not pc.has_index(index_name):
                        pc.create_index(
                            name=index_name,
                            dimension=1000,
                            metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                        st.write(f"Created index '{index_name}'.")
                    index = pc.Index(index_name)
                    for fname in selected_files:
                        file = file_map.get(fname)
                        if file:
                            file.seek(0)
                            json_data = json.load(file)
                            for tab_name, tab_data in json_data.items():
                                if tab_data:
                                    upsert_records(index, tab_name, tab_data)
                    st.success("Upsert completed!")

else:
    st.info("Please upload JSON files to start viewing.")
