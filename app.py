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

def check_if_file_exists_in_pinecone(index, filename):
    """Check if vectors from this file already exist in Pinecone"""
    try:
        # Create a unique identifier based on filename
        file_hash = hashlib.sha256(filename.encode('utf-8')).hexdigest()[:16]
        
        # Try to fetch a vector with this file's identifier
        response = index.query(
            id=file_hash,
            top_k=1,
            include_metadata=True
        )
        
        # If we get results, check if any metadata indicates this file
        for match in response.get('matches', []):
            metadata = match.get('metadata', {})
            if metadata.get('source_file') == filename:
                return True
        return False
    except:
        return False

def upsert_records(index, tab_name, records, filename):
    vectors = []
    for record in records:
        if record is None:
            continue
        record_id = record.get("Reference ID") or record.get("Reference_Id") or record.get("id")
        if not record_id:
            record_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode('utf-8')).hexdigest()
        
        sanitized_metadata = {k.lower().replace(' ', '_'): v for k, v in record.items()}
        # Add source file information to metadata
        sanitized_metadata['source_file'] = filename
        sanitized_metadata['tab_name'] = tab_name
        
        sparse_vector = generate_sparse_vector_from_id(record_id)
        vectors.append({
            "id": record_id,
            "sparse_values": sparse_vector,
            "metadata": sanitized_metadata
        })
    
    if vectors:
        st.write(f"Upserting {len(vectors)} vectors for tab '{tab_name}' from '{filename}'...")
        index.upsert(vectors=vectors)
        st.write(f"Upsert complete for '{tab_name}' from '{filename}'.")
    else:
        st.write(f"No data to upsert for tab '{tab_name}' from '{filename}'.")

st.title("JSON File Manager - Upload and Pinecone Integration")

uploaded_files = st.file_uploader(
    "Upload JSON files", type="json", accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("Please upload no more than 10 files.")
    else:
        st.subheader("Uploaded Files:")
        
        # Simply list the filenames
        for i, file in enumerate(uploaded_files, 1):
            st.write(f"{i}. **{file.name}** ({file.size / 1024:.1f} KB)")
        
        # Option to view file contents (collapsed by default)
        show_details = st.expander("🔍 Click to view file contents (optional)")
        
        with show_details:
            filenames = [file.name for file in uploaded_files]
            selected_files = st.multiselect("Select file(s) to view details", options=filenames)
            
            file_map = {file.name: file for file in uploaded_files}
            
            for fname in selected_files:
                file = file_map.get(fname)
                if file:
                    try:
                        file.seek(0)
                        json_data = json.load(file)
                        st.subheader(f"Contents of {fname}")
                        
                        # Show summary instead of full content
                        st.write(f"**Number of sections:** {len(json_data)}")
                        st.write(f"**Section names:** {', '.join(json_data.keys())}")
                        
                        # Option to show full details
                        if st.checkbox(f"Show full details for {fname}", key=f"details_{fname}"):
                            for tab_name, tab_data in json_data.items():
                                st.write(f"**{tab_name}:** {len(tab_data) if isinstance(tab_data, list) else 'Single entry'}")
                                
                    except Exception as e:
                        st.error(f"Error reading {fname}: {e}")

        # Pinecone upsert section
        st.subheader("📤 Pinecone Integration")
        
        if st.checkbox("Upload to Pinecone"):
            api_key = st.text_input("Enter Pinecone API Key:", type="password")
            index_name = st.text_input("Enter Pinecone Index Name:", value="jeki")
            
            # Check for existing files
            check_existing = st.checkbox("Check for existing files (recommended)", value=True)
            
            if st.button("Process Files"):
                if not api_key:
                    st.error("Please enter Pinecone API key.")
                else:
                    pc = Pinecone(api_key=api_key)
                    
                    # Create index if it doesn't exist
                    if not pc.has_index(index_name):
                        pc.create_index(
                            name=index_name,
                            dimension=1000,
                            metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                        st.write(f"Created index '{index_name}'.")
                    
                    index = pc.Index(index_name)
                    file_map = {file.name: file for file in uploaded_files}
                    
                    processed_count = 0
                    skipped_count = 0
                    
                    for file in uploaded_files:
                        fname = file.name
                        
                        # Check if file already exists in Pinecone
                        if check_existing:
                            if check_if_file_exists_in_pinecone(index, fname):
                                st.warning(f"⚠️ File '{fname}' appears to already exist in Pinecone. Skipping...")
                                skipped_count += 1
                                continue
                        
                        try:
                            file.seek(0)
                            json_data = json.load(file)
                            
                            for tab_name, tab_data in json_data.items():
                                if tab_data:
                                    upsert_records(index, tab_name, tab_data, fname)
                            
                            processed_count += 1
                            st.success(f"✅ Processed '{fname}'")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing '{fname}': {e}")
                    
                    st.success(f"🎉 Upload completed! Processed: {processed_count}, Skipped: {skipped_count}")
else:
    st.info("Please upload JSON files to start.")
