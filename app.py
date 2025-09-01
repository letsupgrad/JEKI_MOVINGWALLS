import streamlit as st
import json
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta
import numpy as np

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
        index.upsert(vectors=vectors)
        st.write(f"Upsert complete for '{tab_name}'. Total records: {len(vectors)}")
        return len(vectors)
    else:
        st.write(f"No data to upsert for tab '{tab_name}'.")
        return 0

def create_performance_chart(data, title):
    """Create performance charts based on data type"""
    if not data or not isinstance(data, (list, dict)):
        return None
    
    fig = None
    
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        df = pd.DataFrame(data)
        
        # Check for common performance metrics
        if 'date' in df.columns or 'Date' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'Date'
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                fig = px.line(df, x=date_col, y=numeric_cols.tolist(), 
                            title=f"{title} - Time Series")
        
        elif any(col in df.columns for col in ['count', 'Count', 'value', 'Value', 'total', 'Total']):
            value_col = next((col for col in ['count', 'Count', 'value', 'Value', 'total', 'Total'] if col in df.columns), None)
            if value_col:
                fig = px.bar(df, y=value_col, title=f"{title} - Distribution")
        
        elif len(df.select_dtypes(include=[np.number]).columns) > 1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # Max 3 columns
            fig = px.scatter_matrix(df, dimensions=numeric_cols.tolist(), 
                                  title=f"{title} - Correlation Matrix")
    
    elif isinstance(data, dict):
        # Create a simple bar chart from dict values if they're numeric
        numeric_items = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        if numeric_items:
            fig = go.Figure(data=[go.Bar(x=list(numeric_items.keys()), y=list(numeric_items.values()))])
            fig.update_layout(title=f"{title} - Summary")
    
    return fig

def search_by_reference_id(json_data_all, reference_id):
    """Search for a specific reference ID across all loaded data"""
    results = {}
    reference_id = str(reference_id).strip()
    
    for filename, file_data in json_data_all.items():
        file_results = {}
        for tab_name, tab_data in file_data.items():
            if isinstance(tab_data, list):
                matching_records = []
                for record in tab_data:
                    if isinstance(record, dict):
                        record_ref_id = str(record.get("Reference ID", record.get("Reference_Id", record.get("id", "")))).strip()
                        if record_ref_id == reference_id:
                            matching_records.append(record)
                if matching_records:
                    file_results[tab_name] = matching_records
            elif isinstance(tab_data, dict):
                record_ref_id = str(tab_data.get("Reference ID", tab_data.get("Reference_Id", tab_data.get("id", "")))).strip()
                if record_ref_id == reference_id:
                    file_results[tab_name] = tab_data
        
        if file_results:
            results[filename] = file_results
    
    return results

def display_detailed_record_info(filename, tab_name, record_data, reference_id):
    """Display detailed information about a specific record"""
    st.subheader(f"📄 File: {filename}")
    st.subheader(f"📋 Tab: {tab_name}")
    st.subheader(f"🔍 Reference ID: {reference_id}")
    
    if isinstance(record_data, list):
        for i, record in enumerate(record_data):
            with st.expander(f"Record {i+1} Details", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Record Information:**")
                    if isinstance(record, dict):
                        for key, value in record.items():
                            st.write(f"• **{key}**: {value}")
                
                with col2:
                    st.write("**Data Visualization:**")
                    # Create chart for this specific record
                    fig = create_performance_chart([record], f"{tab_name} - Record {i+1}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Create a simple info display
                        if isinstance(record, dict):
                            numeric_data = {k: v for k, v in record.items() if isinstance(v, (int, float))}
                            if numeric_data:
                                st.bar_chart(pd.DataFrame([numeric_data]))
    
    elif isinstance(record_data, dict):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Record Information:**")
            for key, value in record_data.items():
                st.write(f"• **{key}**: {value}")
        
        with col2:
            st.write("**Data Visualization:**")
            fig = create_performance_chart(record_data, tab_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                numeric_data = {k: v for k, v in record_data.items() if isinstance(v, (int, float))}
                if numeric_data:
                    st.bar_chart(pd.DataFrame([numeric_data]))

def print_tab(title, data):
    st.header(title)
    if isinstance(data, list):
        if data and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
            st.dataframe(df)
            st.info(f"Total records in {title}: {len(df)}")
            
            # Create enhanced charts
            fig = create_performance_chart(data, title)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
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
        
        # Create chart for dictionary data
        fig = create_performance_chart(data, title)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
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
    
    summary_keywords = ['summary', 'summaries', 'overall', 'performance', 'weekly', 'daily', 'age', 'gender', 'hourly', 'network']
    if any(keyword in query_lower for keyword in summary_keywords):
        return get_summary_categories(json_data)
    
    results = {}
    for filename, file_data in json_data.items():
        for tab_name, tab_data in file_data.items():
            if isinstance(tab_data, list):
                filtered = []
                for record in tab_data:
                    if isinstance(record, dict):
                        if any(query_lower in str(v).lower() for v in record.values()):
                            filtered.append(record)
                if filtered:
                    if filename not in results:
                        results[filename] = {}
                    results[filename][tab_name] = filtered
            elif isinstance(tab_data, dict):
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

# Main Streamlit App
st.set_page_config(page_title="Advanced JSON Data Analyzer", layout="wide")
st.title("🚀 Advanced JSON Data Analyzer with Reference ID Search")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    mode = st.radio("Select Mode:", 
                   ["📁 File Upload & Analysis", "🔍 Reference ID Search", "☁️ Pinecone Operations"])

uploaded_files = st.file_uploader(
    "Upload JSON files", type="json", accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("Please upload no more than 10 files.")
    else:
        filenames = [file.name for file in uploaded_files]
        selected_files = st.multiselect("Select file(s) to analyze", options=filenames, default=filenames)
        file_map = {file.name: file for file in uploaded_files}

        # Load all selected JSON data
        json_data_all = {}
        total_records = 0
        
        for fname in selected_files:
            file = file_map.get(fname)
            if file:
                try:
                    file.seek(0)
                    json_data = json.load(file)
                    json_data_all[fname] = json_data
                    
                    file_record_count = 0
                    for tab_name, tab_data in json_data.items():
                        if isinstance(tab_data, list):
                            file_record_count += len(tab_data)
                        elif isinstance(tab_data, dict):
                            file_record_count += 1
                    
                    total_records += file_record_count
                    
                except Exception as e:
                    st.error(f"Error reading {fname}: {e}")

        # Show different modes based on selection
        if mode == "📁 File Upload & Analysis":
            if selected_files:
                st.metric("📊 Total Records Loaded", total_records)
                
                for fname in selected_files:
                    if fname in json_data_all:
                        st.subheader(f"📄 Contents of {fname}")
                        for tab_name, tab_data in json_data_all[fname].items():
                            print_tab(tab_name, tab_data)
        
        elif mode == "🔍 Reference ID Search":
            st.header("🔍 Search by Reference ID")
            st.info("Enter a Reference ID to get detailed information including file name, charts, and tab details")
            
            reference_id_input = st.text_input("Enter Reference ID:", placeholder="e.g., REF123456")
            
            if reference_id_input and st.button("🔍 Search Reference ID"):
                results = search_by_reference_id(json_data_all, reference_id_input)
                
                if results:
                    st.success(f"✅ Found Reference ID '{reference_id_input}' in {len(results)} file(s)")
                    
                    for filename, file_results in results.items():
                        st.markdown("---")
                        for tab_name, record_data in file_results.items():
                            display_detailed_record_info(filename, tab_name, record_data, reference_id_input)
                else:
                    st.error(f"❌ Reference ID '{reference_id_input}' not found in any loaded files")
                    
                    # Show available Reference IDs for help
                    with st.expander("💡 Available Reference IDs (Sample)"):
                        available_ids = set()
                        for filename, file_data in json_data_all.items():
                            for tab_name, tab_data in file_data.items():
                                if isinstance(tab_data, list):
                                    for record in tab_data[:5]:  # Show first 5
                                        if isinstance(record, dict):
                                            ref_id = record.get("Reference ID", record.get("Reference_Id", record.get("id")))
                                            if ref_id:
                                                available_ids.add(str(ref_id))
                        
                        if available_ids:
                            st.write("Sample Reference IDs found in your data:")
                            for ref_id in list(available_ids)[:10]:  # Show max 10
                                st.code(ref_id)
        
        elif mode == "☁️ Pinecone Operations":
            st.header("☁️ Pinecone Operations")
            
            # General query interface
            st.subheader("📝 Query Data")
            st.info("💡 Use keywords like 'summary', 'performance', 'weekly' to view summary categories")
            query_text = st.text_input("Enter search query:")
            
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
            if st.checkbox("☁️ Upsert to Pinecone"):
                api_key = st.text_input("Pinecone API Key:", type="password")
                index_name = st.text_input("Index Name:", value="jeki")

                if api_key and index_name:
                    try:
                        pc = Pinecone(api_key=api_key)
                        if pc.has_index(index_name):
                            index = pc.Index(index_name)
                            stats = get_pinecone_stats(index)
                            if stats:
                                st.subheader("📊 Current Pinecone Statistics")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Vectors", stats.get('total_vector_count', 0))
                                with col2:
                                    st.metric("Index Fullness", f"{stats.get('index_fullness', 0):.2%}")
                                with col3:
                                    st.metric("Dimension", stats.get('dimension', 'N/A'))
                        else:
                            st.info(f"Index '{index_name}' will be created during upsert.")
                    except Exception as e:
                        st.warning(f"Could not connect to Pinecone: {e}")

                if st.button("🚀 Run Upsert"):
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
                            
                            progress_bar = st.progress(0)
                            total_files = len(selected_files)
                            
                            for i, fname in enumerate(selected_files):
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
                                                count = upsert_records(index, f"{fname}_{tab_name}", [tab_data])
                                                total_upserted += count
                                
                                progress_bar.progress((i + 1) / total_files)
                            
                            st.success(f"✅ Upsert completed! Total records uploaded: {total_upserted}")
                            
                            # Show updated stats
                            updated_stats = get_pinecone_stats(index)
                            if updated_stats:
                                st.subheader("📊 Updated Pinecone Statistics")
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
    st.info("👆 Please upload JSON files to start analyzing your data")
    st.markdown("""
    ### Features Available:
    - 📁 **File Upload & Analysis**: View and analyze your JSON files with interactive charts
    - 🔍 **Reference ID Search**: Search for specific records by Reference ID with detailed views
    - ☁️ **Pinecone Operations**: Upload data to Pinecone with real-time statistics
    - 📊 **Interactive Charts**: Automatic chart generation based on your data structure
    - 🎯 **Smart Queries**: Search for summaries, performance data, and more
    """)
