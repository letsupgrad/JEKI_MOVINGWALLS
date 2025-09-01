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
        st.write(f"Upsert complete for '{tab_name}'. Total records: {len(vectors)}")
        return len(vectors)
    else:
        st.write(f"No data to upsert for tab '{tab_name}'.")
        return 0

def print_tab(title, data):
    st.header(title)
    if isinstance(data, list):
        # Detect if data is list of dicts (tabular)
        if data and all(isinstance(item, dict) for item in data):
            # Display as a dataframe
            df = pd.DataFrame(data)
            st.dataframe(df)
            # Show record count
            st.info(f"Total records in {title}: {len(df)}")
            # Create simple bar chart for numeric columns if any
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                st.subheader("Charts")
                for col in numeric_cols:
                    st.bar_chart(df[col])
        else:
            st.info(f"Total entries in {title}: {len(data)}")
            for i, entry in enumerate(data, start=1):
                with st.expander(f"Entry {i}"):
                    if isinstance(entry, dict):
                        for key, value in entry.items():
                            st.write(f"**{key}**: {value}")
                    else:
                        st.write(entry)
    elif isinstance(data, dict):
        st.info(f"Dictionary entries in {title}: {len(data)}")
        for key, value in data.items():
            st.write(f"**{key}**: {value}")
    else:
        st.write(data)

def get_summary_categories(json_data):
    """Get predefined summary categories from the data"""
    summary_categories = [
        "Overall Performance Summary",
        "Weekly Summary", 
        "Daily Summary",
        "Overall Age and Gender",
        "Overall Hourly",
        "Network Summary"
    ]
    
    results = {}
    for filename, file_data in json_data.items():
        for category in summary_categories:
            if category in file_data:
                if filename not in results:
                    results[filename] = {}
                results[filename][category] = file_data[category]
    
    return results

def query_data(json_data, query):
    """Enhanced query: check for summary request or filter records containing query substring"""
    query_lower = query.lower()
    
    # Check if user is asking for summary
    summary_keywords = ['summary', 'summaries', 'overall', 'performance', 'weekly', 'daily', 'age', 'gender', 'hourly', 'network']
    if any(keyword in query_lower for keyword in summary_keywords):
        return get_summary_categories(json_data)
    
    # Regular substring search
    results = {}
    for filename, file_data in json_data.items():
        for tab_name, tab_data in file_data.items():
            if isinstance(tab_data, list):
                filtered = []
                for record in tab_data:
                    if isinstance(record, dict):
                        # Check if any string field contains query substring
                        if any(query_lower in str(v).lower() for v in record.values()):
                            filtered.append(record)
                if filtered:
                    if filename not in results:
                        results[filename] = {}
                    results[filename][tab_name] = filtered
            elif isinstance(tab_data, dict):
                # For dict data, check if query in any value
                if any(query_lower in str(v).lower() for v in tab_data.values()):
                    if filename not in results:
                        results[filename] = {}
                    results[filename][tab_name] = tab_data
    
    return results

def get_pinecone_stats(index):
    """Get statistics from Pinecone index"""
    try:
        stats = index.describe_index_stats()
        return stats
    except Exception as e:
        st.error(f"Error getting Pinecone stats: {e}")
        return None

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
        total_records = 0
        
        for fname in selected_files:
            file = file_map.get(fname)
            if file:
                try:
                    file.seek(0)
                    json_data = json.load(file)
                    json_data_all[fname] = json_data
                    st.subheader(f"Contents of {fname}")
                    
                    # Count records in this file
                    file_record_count = 0
                    for tab_name, tab_data in json_data.items():
                        if isinstance(tab_data, list):
                            file_record_count += len(tab_data)
                        elif isinstance(tab_data, dict):
                            file_record_count += 1
                        print_tab(tab_name, tab_data)
                    
                    total_records += file_record_count
                    st.success(f"File {fname} loaded with {file_record_count} total records")
                    
                except Exception as e:
                    st.error(f"Error reading {fname}: {e}")

        # Show total records across all files
        if selected_files:
            st.metric("Total Records Loaded", total_records)

        # Query interface on combined JSON data from selected files
        if selected_files:
            st.subheader("Query data across selected files")
            st.info("💡 Tip: Use keywords like 'summary', 'performance', 'weekly', 'daily', etc. to view summary categories")
            query_text = st.text_input("Enter search query (case insensitive substring match)")
            
            if query_text:
                results = query_data(json_data_all, query_text)
                if results:
                    st.write(f"Query results for '{query_text}':")
                    result_count = 0
                    for filename, file_results in results.items():
                        st.subheader(f"Results from {filename}")
                        for tab_name, tab_results in file_results.items():
                            if isinstance(tab_results, list):
                                result_count += len(tab_results)
                            else:
                                result_count += 1
                            print_tab(tab_name, tab_results)
                    st.success(f"Found {result_count} matching records")
                else:
                    st.write("No matching records found.")

        # Pinecone upsert section
        if st.checkbox("Upsert selected data to Pinecone"):
            api_key = st.text_input("Enter Pinecone API Key:", type="password")
            index_name = st.text_input("Enter Pinecone Index Name:", value="jeki")

            # Show current Pinecone stats if API key is provided
            if api_key and index_name:
                try:
                    pc = Pinecone(api_key=api_key)
                    if pc.has_index(index_name):
                        index = pc.Index(index_name)
                        stats = get_pinecone_stats(index)
                        if stats:
                            st.subheader("Current Pinecone Index Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Vectors", stats.get('total_vector_count', 0))
                            with col2:
                                st.metric("Index Fullness", f"{stats.get('index_fullness', 0):.2%}")
                            with col3:
                                st.metric("Dimension", stats.get('dimension', 'N/A'))
                    else:
                        st.info(f"Index '{index_name}' does not exist. It will be created during upsert.")
                except Exception as e:
                    st.warning(f"Could not connect to Pinecone: {e}")

            if st.button("Run Upsert"):
                if not (api_key and selected_files):
                    st.error("Please select files and enter API key.")
                else:
                    try:
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
                        total_upserted = 0
                        
                        for fname in selected_files:
                            file = file_map.get(fname)
                            if file:
                                file.seek(0)
                                json_data = json.load(file)
                                for tab_name, tab_data in json_data.items():
                                    if tab_data:
                                        if isinstance(tab_data, list):
                                            count = upsert_records(index, f"{fname}_{tab_name}", tab_data)
                                            total_upserted += count
                                        elif isinstance(tab_data, dict):
                                            # Convert single dict to list for upsert
                                            count = upsert_records(index, f"{fname}_{tab_name}", [tab_data])
                                            total_upserted += count
                        
                        st.success(f"Upsert completed! Total records uploaded: {total_upserted}")
                        
                        # Show updated stats
                        updated_stats = get_pinecone_stats(index)
                        if updated_stats:
                            st.subheader("Updated Pinecone Index Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Vectors", updated_stats.get('total_vector_count', 0))
                            with col2:
                                st.metric("Index Fullness", f"{updated_stats.get('index_fullness', 0):.2%}")
                            with col3:
                                st.metric("Dimension", updated_stats.get('dimension', 'N/A'))
                                
                    except Exception as e:
                        st.error(f"Error during upsert: {e}")

else:
    st.info("Please upload JSON files to start viewing.")
