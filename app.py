import streamlit as st
import json
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional, Union
import time

# Configuration and constants
MAX_FILES = 15
MAX_SAMPLE_IDS = 20
PROGRESS_UPDATE_INTERVAL = 0.1

# Enhanced data processing functions
def generate_sparse_vector_from_id(record_id: str, num_indices: int = 5, max_index: int = 1000) -> Dict[str, List]:
    """Generate deterministic sparse vector from record ID using SHA-256 hash"""
    h = hashlib.sha256(record_id.encode('utf-8')).hexdigest()
    seen_indices = set()
    
    # Generate unique indices
    for i in range(num_indices * 4):
        if len(seen_indices) >= num_indices:
            break
        index = int(h[i*2: i*2+4], 16) % max_index
        seen_indices.add(index)
    
    indices = sorted(list(seen_indices))
    values = []
    
    # Generate values from hash
    for i in range(len(indices)):
        value = 0.1 + (int(h[i*4: i*4+4], 16) % 900) / 1000.0
        values.append(round(value, 4))
    
    return {"indices": indices, "values": values}

def sanitize_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metadata for Pinecone compatibility"""
    sanitized = {}
    for k, v in record.items():
        # Convert key to lowercase and replace spaces with underscores
        clean_key = str(k).lower().replace(' ', '_').replace('-', '_')
        
        # Handle different value types
        if isinstance(v, (str, int, float, bool)):
            sanitized[clean_key] = v
        elif isinstance(v, (list, dict)):
            sanitized[clean_key] = str(v)  # Convert complex types to string
        else:
            sanitized[clean_key] = str(v)
    
    return sanitized

def upsert_records(index, tab_name: str, records: List[Dict], batch_size: int = 100) -> int:
    """Upsert records to Pinecone in batches with error handling"""
    if not records:
        st.warning(f"No data to upsert for tab '{tab_name}'.")
        return 0
    
    vectors = []
    successful_count = 0
    
    for record in records:
        if record is None:
            continue
            
        try:
            # Get record ID with fallback options
            record_id = (record.get("Reference ID") or 
                        record.get("Reference_Id") or 
                        record.get("id") or
                        hashlib.sha256(json.dumps(record, sort_keys=True).encode('utf-8')).hexdigest())
            
            # Sanitize metadata
            sanitized_metadata = sanitize_metadata(record)
            
            # Generate sparse vector
            sparse_vector = generate_sparse_vector_from_id(str(record_id))
            
            vectors.append({
                "id": str(record_id),
                "sparse_values": sparse_vector,
                "metadata": sanitized_metadata
            })
            
        except Exception as e:
            st.warning(f"Skipping record due to error: {e}")
            continue
    
    if not vectors:
        st.warning(f"No valid vectors generated for tab '{tab_name}'.")
        return 0
    
    # Upsert in batches
    progress_container = st.empty()
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        try:
            index.upsert(vectors=batch)
            successful_count += len(batch)
            
            # Update progress
            progress_container.text(f"Batch {batch_num}/{total_batches} uploaded for '{tab_name}'")
            time.sleep(PROGRESS_UPDATE_INTERVAL)
            
        except Exception as e:
            st.error(f"Error upserting batch {batch_num} for '{tab_name}': {e}")
    
    progress_container.empty()
    st.success(f"✅ Upserted {successful_count}/{len(vectors)} records for '{tab_name}'")
    return successful_count

def create_enhanced_performance_chart(data: Union[List, Dict], title: str) -> Optional[go.Figure]:
    """Create enhanced performance charts with better error handling and more chart types"""
    if not data or not isinstance(data, (list, dict)):
        return None
    
    try:
        fig = None
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            
            # Time series detection
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if numeric_cols:
                        fig = px.line(df, x=date_col, y=numeric_cols[:3],  # Max 3 lines
                                    title=f"{title} - Time Series Analysis",
                                    template="plotly_white")
                except Exception:
                    pass
            
            # Performance metrics detection
            if fig is None:
                metric_cols = [col for col in df.columns if any(metric in col.lower() 
                              for metric in ['count', 'value', 'total', 'score', 'rate', 'percentage'])]
                if metric_cols:
                    metric_col = metric_cols[0]
                    if df[metric_col].dtype in ['int64', 'float64']:
                        fig = px.bar(df.head(20), y=metric_col, 
                                   title=f"{title} - Performance Distribution",
                                   template="plotly_white")
            
            # Correlation analysis for multiple numeric columns
            if fig is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    corr_cols = numeric_cols[:4]  # Max 4 columns for readability
                    fig = px.scatter_matrix(df, dimensions=corr_cols.tolist(),
                                          title=f"{title} - Correlation Analysis",
                                          template="plotly_white")
        
        elif isinstance(data, dict):
            # Enhanced dictionary visualization
            numeric_items = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            if numeric_items:
                # Create different chart types based on data characteristics
                if len(numeric_items) <= 5:
                    fig = go.Figure(data=[
                        go.Bar(x=list(numeric_items.keys()), 
                               y=list(numeric_items.values()),
                               marker_color='lightblue')
                    ])
                else:
                    # Use horizontal bar for many items
                    fig = go.Figure(data=[
                        go.Bar(x=list(numeric_items.values()),
                               y=list(numeric_items.keys()),
                               orientation='h',
                               marker_color='lightcoral')
                    ])
                
                fig.update_layout(title=f"{title} - Summary Dashboard",
                                template="plotly_white")
        
        # Add common styling
        if fig:
            fig.update_layout(
                showlegend=True,
                height=400,
                margin=dict(l=50, r=50, t=80, b=50)
            )
        
        return fig
    
    except Exception as e:
        st.warning(f"Could not create chart for {title}: {e}")
        return None

def search_by_reference_id(json_data_all: Dict, reference_id: str) -> Dict:
    """Enhanced search with better matching and case insensitive search"""
    results = {}
    reference_id = str(reference_id).strip().lower()
    
    for filename, file_data in json_data_all.items():
        file_results = {}
        
        for tab_name, tab_data in file_data.items():
            if isinstance(tab_data, list):
                matching_records = []
                for record in tab_data:
                    if isinstance(record, dict):
                        # Check multiple possible ID fields
                        id_fields = ["Reference ID", "Reference_Id", "id", "ID", "ref_id", "reference_id"]
                        for field in id_fields:
                            record_ref_id = str(record.get(field, "")).strip().lower()
                            if record_ref_id == reference_id:
                                matching_records.append(record)
                                break
                
                if matching_records:
                    file_results[tab_name] = matching_records
                    
            elif isinstance(tab_data, dict):
                # Check single record
                id_fields = ["Reference ID", "Reference_Id", "id", "ID", "ref_id", "reference_id"]
                for field in id_fields:
                    record_ref_id = str(tab_data.get(field, "")).strip().lower()
                    if record_ref_id == reference_id:
                        file_results[tab_name] = tab_data
                        break
        
        if file_results:
            results[filename] = file_results
    
    return results

def display_enhanced_record_info(filename: str, tab_name: str, record_data: Union[List, Dict], reference_id: str):
    """Enhanced record display with better formatting and additional insights"""
    
    # Header with better formatting
    st.markdown(f"""
    ### 📄 **File:** `{filename}`
    ### 📋 **Tab:** `{tab_name}`  
    ### 🔍 **Reference ID:** `{reference_id}`
    """)
    
    if isinstance(record_data, list):
        st.info(f"Found {len(record_data)} matching record(s)")
        
        for i, record in enumerate(record_data):
            with st.expander(f"📊 Record {i+1} Details", expanded=i==0):  # Expand first record by default
                
                # Create tabs for better organization
                tab1, tab2, tab3 = st.tabs(["📋 Data", "📈 Visualization", "🔍 Analysis"])
                
                with tab1:
                    if isinstance(record, dict):
                        # Create a nice table format
                        record_df = pd.DataFrame(list(record.items()), columns=['Field', 'Value'])
                        st.dataframe(record_df, use_container_width=True)
                        
                        # Show data types and statistics
                        with st.expander("📊 Field Statistics"):
                            numeric_fields = {k: v for k, v in record.items() if isinstance(v, (int, float))}
                            if numeric_fields:
                                st.write("**Numeric Fields:**")
                                for field, value in numeric_fields.items():
                                    st.metric(field, value)
                
                with tab2:
                    fig = create_enhanced_performance_chart([record], f"{tab_name} - Record {i+1}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback visualization
                        if isinstance(record, dict):
                            numeric_data = {k: v for k, v in record.items() if isinstance(v, (int, float))}
                            if numeric_data and len(numeric_data) > 0:
                                st.bar_chart(pd.DataFrame([numeric_data]))
                            else:
                                st.info("No numeric data available for visualization")
                
                with tab3:
                    if isinstance(record, dict):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Record Summary:**")
                            st.write(f"• Total fields: {len(record)}")
                            numeric_count = sum(1 for v in record.values() if isinstance(v, (int, float)))
                            st.write(f"• Numeric fields: {numeric_count}")
                            text_count = sum(1 for v in record.values() if isinstance(v, str))
                            st.write(f"• Text fields: {text_count}")
                        
                        with col2:
                            st.write("**Field Types:**")
                            type_counts = {}
                            for value in record.values():
                                type_name = type(value).__name__
                                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                            
                            for type_name, count in type_counts.items():
                                st.write(f"• {type_name}: {count}")
    
    elif isinstance(record_data, dict):
        # Single record display
        tab1, tab2, tab3 = st.tabs(["📋 Data", "📈 Visualization", "🔍 Analysis"])
        
        with tab1:
            record_df = pd.DataFrame(list(record_data.items()), columns=['Field', 'Value'])
            st.dataframe(record_df, use_container_width=True)
        
        with tab2:
            fig = create_enhanced_performance_chart(record_data, tab_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                numeric_data = {k: v for k, v in record_data.items() if isinstance(v, (int, float))}
                if numeric_data:
                    st.bar_chart(pd.DataFrame([numeric_data]))
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Record Summary:**")
                st.write(f"• Total fields: {len(record_data)}")
                numeric_count = sum(1 for v in record_data.values() if isinstance(v, (int, float)))
                st.write(f"• Numeric fields: {numeric_count}")
            
            with col2:
                st.write("**Field Types:**")
                type_counts = {}
                for value in record_data.values():
                    type_name = type(value).__name__
                    type_counts[type_name] = type_counts.get(type_name, 0) + 1
                
                for type_name, count in type_counts.items():
                    st.write(f"• {type_name}: {count}")

def print_enhanced_tab(title: str, data: Union[List, Dict]):
    """Enhanced tab display with better formatting and statistics"""
    st.header(f"📊 {title}")
    
    if isinstance(data, list):
        if data and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
            
            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📋 Total Records", len(df))
            with col2:
                st.metric("📑 Total Fields", len(df.columns))
            with col3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                st.metric("🔢 Numeric Fields", len(numeric_cols))
            
            # Display data with pagination for large datasets
            if len(df) > 1000:
                st.warning(f"Large dataset detected ({len(df)} rows). Showing first 1000 rows.")
                st.dataframe(df.head(1000), use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
            
            # Enhanced visualization
            fig = create_enhanced_performance_chart(data, title)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"📄 Total entries in {title}: {len(data)}")
            # Show sample entries for non-dictionary lists
            if data:
                with st.expander("Sample Entries"):
                    for i, entry in enumerate(data[:5], start=1):  # Show first 5
                        st.write(f"**Entry {i}:** {entry}")
    
    elif isinstance(data, dict):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Dictionary Entries", len(data))
        with col2:
            numeric_entries = sum(1 for v in data.values() if isinstance(v, (int, float)))
            st.metric("🔢 Numeric Entries", numeric_entries)
        
        # Display as table
        if data:
            data_df = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])
            st.dataframe(data_df, use_container_width=True)
        
        # Create visualization
        fig = create_enhanced_performance_chart(data, title)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(data)

def get_summary_categories(json_data: Dict) -> Dict:
    """Enhanced summary category detection"""
    summary_categories = [
        "Overall Performance Summary",
        "Weekly Summary", 
        "Daily Summary",
        "Overall Age and Gender",
        "Overall Hourly",
        "Network Summary",
        "Performance Overview",
        "Analytics Summary",
        "Dashboard Data"
    ]
    
    results = {}
    for filename, file_data in json_data.items():
        for category in summary_categories:
            # Case-insensitive search
            for key in file_data.keys():
                if key.lower() == category.lower():
                    if filename not in results:
                        results[filename] = {}
                    results[filename][category] = file_data[key]
                    break
    
    return results

def enhanced_query_data(json_data: Dict, query: str) -> Dict:
    """Enhanced query with better search capabilities"""
    query_lower = query.lower().strip()
    
    # Summary keywords detection
    summary_keywords = ['summary', 'summaries', 'overall', 'performance', 'weekly', 
                       'daily', 'age', 'gender', 'hourly', 'network', 'analytics', 'dashboard']
    
    if any(keyword in query_lower for keyword in summary_keywords):
        return get_summary_categories(json_data)
    
    results = {}
    for filename, file_data in json_data.items():
        for tab_name, tab_data in file_data.items():
            if isinstance(tab_data, list):
                filtered = []
                for record in tab_data:
                    if isinstance(record, dict):
                        # Search in all fields (keys and values)
                        if any(query_lower in str(k).lower() or query_lower in str(v).lower() 
                              for k, v in record.items()):
                            filtered.append(record)
                
                if filtered:
                    if filename not in results:
                        results[filename] = {}
                    results[filename][tab_name] = filtered
            
            elif isinstance(tab_data, dict):
                # Search in dictionary keys and values
                if any(query_lower in str(k).lower() or query_lower in str(v).lower() 
                      for k, v in tab_data.items()):
                    if filename not in results:
                        results[filename] = {}
                    results[filename][tab_name] = tab_data
    
    return results

def get_enhanced_pinecone_stats(index) -> Optional[Dict]:
    """Get enhanced Pinecone statistics with error handling"""
    try:
        stats = index.describe_index_stats()
        return stats
    except Exception as e:
        st.error(f"Error getting Pinecone statistics: {e}")
        return None

def extract_sample_reference_ids(json_data_all: Dict, max_samples: int = MAX_SAMPLE_IDS) -> List[str]:
    """Extract sample reference IDs from loaded data"""
    available_ids = set()
    
    for filename, file_data in json_data_all.items():
        for tab_name, tab_data in file_data.items():
            if isinstance(tab_data, list):
                for record in tab_data:
                    if isinstance(record, dict):
                        id_fields = ["Reference ID", "Reference_Id", "id", "ID", "ref_id"]
                        for field in id_fields:
                            ref_id = record.get(field)
                            if ref_id:
                                available_ids.add(str(ref_id))
                                if len(available_ids) >= max_samples:
                                    return list(available_ids)
            elif isinstance(tab_data, dict):
                id_fields = ["Reference ID", "Reference_Id", "id", "ID", "ref_id"]
                for field in id_fields:
                    ref_id = tab_data.get(field)
                    if ref_id:
                        available_ids.add(str(ref_id))
                        if len(available_ids) >= max_samples:
                            return list(available_ids)
    
    return list(available_ids)

# Main Streamlit Application
def main():
    st.set_page_config(
        page_title="Advanced JSON Data Analyzer", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Advanced JSON Data Analyzer with Pinecone Integration")
    st.markdown("---")
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("🧭 Navigation")
        st.markdown("Select your preferred analysis mode:")
        
        mode = st.radio(
            "Choose Mode:", 
            ["📁 File Upload & Analysis", "🔍 Reference ID Search", "☁️ Pinecone Operations"],
            help="Select the type of analysis you want to perform"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Features:**
        - 📊 Interactive data visualization
        - 🔍 Smart reference ID search
        - ☁️ Pinecone vector database integration
        - 📈 Enhanced performance charts
        - 🎯 Intelligent query system
        """)
    
    # File upload section
    st.header("📤 File Upload")
    uploaded_files = st.file_uploader(
        "Upload JSON files for analysis", 
        type="json", 
        accept_multiple_files=True,
        help=f"You can upload up to {MAX_FILES} JSON files at once"
    )
    
    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.error(f"⚠️ Please upload no more than {MAX_FILES} files.")
            return
        
        # File selection
        filenames = [file.name for file in uploaded_files]
        selected_files = st.multiselect(
            "📋 Select files to analyze:", 
            options=filenames, 
            default=filenames,
            help="Choose which uploaded files you want to analyze"
        )
        
        if not selected_files:
            st.warning("Please select at least one file to analyze.")
            return
        
        file_map = {file.name: file for file in uploaded_files}
        
        # Load and process selected files
        with st.spinner("🔄 Loading and processing files..."):
            json_data_all, total_records = load_json_files(selected_files, file_map)
        
        if not json_data_all:
            st.error("No valid JSON files could be loaded.")
            return
        
        # Display loading summary
        st.success(f"✅ Successfully loaded {len(selected_files)} file(s)")
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Records", total_records)
        with col2:
            st.metric("📁 Files Loaded", len(json_data_all))
        with col3:
            total_tabs = sum(len(file_data) for file_data in json_data_all.values())
            st.metric("📋 Total Tabs", total_tabs)
        
        st.markdown("---")
        
        # Mode-specific functionality
        if mode == "📁 File Upload & Analysis":
            handle_file_analysis_mode(json_data_all, selected_files)
        
        elif mode == "🔍 Reference ID Search":
            handle_reference_search_mode(json_data_all)
        
        elif mode == "☁️ Pinecone Operations":
            handle_pinecone_operations_mode(json_data_all, selected_files, file_map)
    
    else:
        display_welcome_message()

def load_json_files(selected_files: List[str], file_map: Dict) -> tuple:
    """Load and validate JSON files with error handling"""
    json_data_all = {}
    total_records = 0
    
    for fname in selected_files:
        file = file_map.get(fname)
        if file:
            try:
                file.seek(0)
                json_data = json.load(file)
                json_data_all[fname] = json_data
                
                # Count records
                file_record_count = 0
                for tab_name, tab_data in json_data.items():
                    if isinstance(tab_data, list):
                        file_record_count += len(tab_data)
                    elif isinstance(tab_data, dict):
                        file_record_count += 1
                
                total_records += file_record_count
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON in {fname}: {e}")
            except Exception as e:
                st.error(f"❌ Error reading {fname}: {e}")
    
    return json_data_all, total_records

def handle_file_analysis_mode(json_data_all: Dict, selected_files: List[str]):
    """Handle file analysis mode"""
    st.header("📁 File Analysis & Visualization")
    
    for fname in selected_files:
        if fname in json_data_all:
            with st.expander(f"📄 {fname}", expanded=len(selected_files) == 1):
                for tab_name, tab_data in json_data_all[fname].items():
                    print_enhanced_tab(tab_name, tab_data)

def handle_reference_search_mode(json_data_all: Dict):
    """Handle reference ID search mode"""
    st.header("🔍 Advanced Reference ID Search")
    st.info("💡 Enter a Reference ID to get comprehensive information including file location, data visualization, and detailed analysis")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        reference_id_input = st.text_input(
            "🔍 Enter Reference ID:", 
            placeholder="e.g., REF123456, ID_001, etc.",
            help="Enter the exact Reference ID you want to search for"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if reference_id_input and search_button:
        with st.spinner("🔄 Searching for Reference ID..."):
            results = search_by_reference_id(json_data_all, reference_id_input)
        
        if results:
            st.success(f"✅ Found Reference ID '{reference_id_input}' in {len(results)} file(s)")
            
            for filename, file_results in results.items():
                st.markdown("---")
                for tab_name, record_data in file_results.items():
                    display_enhanced_record_info(filename, tab_name, record_data, reference_id_input)
        else:
            st.error(f"❌ Reference ID '{reference_id_input}' not found in any loaded files")
            
            # Show available Reference IDs
            with st.expander("💡 Available Reference IDs (Sample)"):
                available_ids = extract_sample_reference_ids(json_data_all)
                
                if available_ids:
                    st.write("**Sample Reference IDs found in your data:**")
                    
                    # Display in columns for better layout
                    cols = st.columns(3)
                    for i, ref_id in enumerate(available_ids):
                        with cols[i % 3]:
                            if st.button(f"🔗 {ref_id}", key=f"ref_{i}"):
                                st.experimental_rerun()  # Trigger search with this ID
                else:
                    st.warning("No Reference IDs found in the loaded data.")

def handle_pinecone_operations_mode(json_data_all: Dict, selected_files: List[str], file_map: Dict):
    """Handle Pinecone operations mode"""
    st.header("☁️ Pinecone Vector Database Operations")
    
    # Query interface
    st.subheader("📝 Smart Query System")
    st.info("💡 Use keywords like 'summary', 'performance', 'weekly' to find summary data, or enter any search term")
    
    query_text = st.text_input("🔍 Enter search query:", placeholder="e.g., summary, performance, specific values...")
    
    if query_text:
        with st.spinner("🔄 Searching data..."):
            results = enhanced_query_data(json_data_all, query_text)
        
        if results:
            result_count = sum(
                len(tab_results) if isinstance(tab_results, list) else 1 
                for file_results in results.values() 
                for tab_results in file_results.values()
            )
            
            st.success(f"📊 Found {result_count} matching record(s) for query: '{query_text}'")
            
            for filename, file_results in results.items():
                st.subheader(f"📄 Results from {filename}")
                for tab_name, tab_results in file_results.items():
                    print_enhanced_tab(f"{tab_name} (Query Results)", tab_results)
        else:
            st.warning("No matching records found for your query.")
    
    st.markdown("---")
    
    # Pinecone integration
    st.subheader("☁️ Pinecone Vector Database Integration")
    
    pinecone_expander = st.expander("🔧 Pinecone Configuration", expanded=False)
    with pinecone_exp
