import streamlit as st
from pinecone import Pinecone
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import time

# === Page Configuration ===
st.set_page_config(
    page_title="Campaign Data Viewer",
    page_icon="📊",
    layout="wide"
)

# === Initialize Session State ===
if 'pinecone_connected' not in st.session_state:
    st.session_state.pinecone_connected = False

# === Sidebar Configuration ===
st.sidebar.title("🔧 Configuration")

# Pinecone API Key Input
api_key = st.sidebar.text_input(
    "Pinecone API Key", 
    value="pcsk_3wbxiS_JFsW8uFyumkQ2oMD5FkfjKJPV5kYkiDwX1T15tg2HtFSn4ioZEeVpsSV6V1DK7s",
    type="password"
)

index_name = st.sidebar.text_input("Index Name", value="campaign")

# === Connect to Pinecone ===
@st.cache_resource
def connect_to_pinecone(api_key, index_name):
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        model = SentenceTransformer("intfloat/multilingual-e5-large")
        return pc, index, model
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None, None, None

# === Available Tab Details ===
AVAILABLE_TABS = {
    "Billboard_Details_Tab": [
        "Overall Performance Summary",
        "Weekly Summary", 
        "Daily Summary",
        "Overall Age and Gender",
        "Overall Hourly",
        "Network Summary"
    ],
    "Campaign_Details_Tab": [
        "Campaign Overview",
        "Budget Allocation",
        "Target Audience",
        "Performance Metrics",
        "ROI Analysis"
    ],
    "Analytics_Tab": [
        "Traffic Analysis",
        "Conversion Tracking",
        "User Behavior",
        "Geographic Data",
        "Device Analytics"
    ]
}

# === Main App ===
def main():
    st.title("📊 Campaign Data Individual Record Viewer")
    st.markdown("---")
    
    # Connect to Pinecone
    if api_key and index_name:
        pc, index, model = connect_to_pinecone(api_key, index_name)
        if index is not None:
            st.session_state.pinecone_connected = True
            st.sidebar.success("✅ Connected to Pinecone")
            
            # === Tab Selection ===
            st.header("🎯 Select Tab Details to View")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_tab = st.selectbox(
                    "Choose Tab Category:",
                    list(AVAILABLE_TABS.keys()),
                    help="Select which tab category you want to explore"
                )
            
            with col2:
                selected_detail = st.selectbox(
                    "Choose Specific Detail:",
                    AVAILABLE_TABS[selected_tab],
                    help="Select the specific detail you want to view"
                )
            
            st.markdown("---")
            
            # === Search Methods ===
            st.header("🔍 Find Records")
            
            search_method = st.radio(
                "Search Method:",
                ["Search by Content", "View by Record Index", "Browse All Records"],
                horizontal=True
            )
            
            if search_method == "Search by Content":
                search_by_content(index, model, selected_tab, selected_detail)
            elif search_method == "View by Record Index":
                view_by_index(index, selected_tab, selected_detail)
            else:
                browse_all_records(index, selected_tab, selected_detail)
        else:
            st.error("❌ Unable to connect to Pinecone. Please check your credentials.")
    else:
        st.warning("⚠️ Please enter Pinecone API key and index name in the sidebar.")

# === Search by Content ===
def search_by_content(index, model, selected_tab, selected_detail):
    st.subheader(f"🔍 Search Records for: {selected_tab} → {selected_detail}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter search terms:",
            placeholder="e.g., Dell campaign, April data, performance metrics..."
        )
    
    with col2:
        num_results = st.number_input("Results", min_value=1, max_value=20, value=5)
    
    if search_query:
        with st.spinner("🔍 Searching records..."):
            try:
                # Generate query embedding
                query_embedding = model.encode(search_query).tolist()
                
                # Search Pinecone
                results = index.query(
                    vector=query_embedding,
                    top_k=num_results,
                    include_metadata=True
                )
                
                if results['matches']:
                    st.success(f"✅ Found {len(results['matches'])} matching records")
                    
                    # Display results
                    for i, match in enumerate(results['matches']):
                        with st.expander(f"📊 Record #{match['id']} - Similarity: {match['score']:.4f}"):
                            display_record_details(match['metadata'], selected_tab, selected_detail, match['id'])
                else:
                    st.warning("❌ No matching records found")
                    
            except Exception as e:
                st.error(f"❌ Search Error: {e}")

# === View by Index ===
def view_by_index(index, selected_tab, selected_detail):
    st.subheader(f"📋 View Record by Index: {selected_tab} → {selected_detail}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        record_index = st.number_input(
            "Enter Record Index:",
            min_value=0,
            value=0,
            help="Enter the row index number of the record you want to view"
        )
    
    with col2:
        if st.button("🔍 View Record", type="primary"):
            with st.spinner("📊 Fetching record..."):
                try:
                    response = index.fetch(ids=[str(record_index)])
                    
                    if str(record_index) in response['vectors']:
                        record = response['vectors'][str(record_index)]
                        st.success(f"✅ Record found!")
                        display_record_details(record['metadata'], selected_tab, selected_detail, record_index)
                    else:
                        st.error(f"❌ Record with index {record_index} not found")
                        
                except Exception as e:
                    st.error(f"❌ Fetch Error: {e}")

# === Browse All Records ===
def browse_all_records(index, selected_tab, selected_detail):
    st.subheader(f"📖 Browse Records: {selected_tab} → {selected_detail}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        page_size = st.number_input("Records per page:", min_value=1, max_value=20, value=5)
    
    with col2:
        page_number = st.number_input("Page number:", min_value=1, value=1)
    
    with col3:
        if st.button("📖 Load Page", type="primary"):
            start_idx = (page_number - 1) * page_size
            end_idx = start_idx + page_size
            
            with st.spinner("📊 Loading records..."):
                try:
                    record_ids = [str(i) for i in range(start_idx, end_idx)]
                    response = index.fetch(ids=record_ids)
                    
                    st.success(f"✅ Loaded page {page_number}")
                    
                    for i in range(start_idx, end_idx):
                        record_id = str(i)
                        if record_id in response['vectors']:
                            record = response['vectors'][record_id]
                            with st.expander(f"📊 Record #{i}"):
                                display_record_details(record['metadata'], selected_tab, selected_detail, i)
                        
                except Exception as e:
                    st.error(f"❌ Browse Error: {e}")

# === Display Record Details ===
def display_record_details(metadata, selected_tab, selected_detail, record_id):
    """Display detailed record information based on selected tab and detail"""
    
    # Header
    st.markdown(f"### 📋 Record ID: {record_id}")
    st.markdown(f"**Tab:** {selected_tab} → **Detail:** {selected_detail}")
    
    # Create tabs for organized display
    tab1, tab2, tab3 = st.tabs(["🎯 Filtered View", "📊 All Data", "🔧 Raw JSON"])
    
    with tab1:
        st.markdown(f"#### 🎯 {selected_detail} Details")
        
        # Filter data based on selected detail
        filtered_data = filter_data_by_detail(metadata, selected_tab, selected_detail)
        
        if filtered_data:
            # Display as metrics if numerical
            if len(filtered_data) <= 4:
                cols = st.columns(len(filtered_data))
                for i, (key, value) in enumerate(filtered_data.items()):
                    with cols[i]:
                        st.metric(key, value)
            else:
                # Display as key-value pairs
                for key, value in filtered_data.items():
                    st.write(f"**{key}:** {value}")
        else:
            st.info(f"ℹ️ No specific data found for '{selected_detail}'")
    
    with tab2:
        st.markdown("#### 📊 Complete Record Data")
        
        # Group data by categories
        categorized_data = categorize_metadata(metadata)
        
        for category, fields in categorized_data.items():
            if fields:
                st.markdown(f"**{category}:**")
                for key, value in fields.items():
                    st.write(f"• **{key}:** {value}")
                st.markdown("---")
    
    with tab3:
        st.markdown("#### 🔧 Raw Data (JSON)")
        st.json(metadata)

# === Filter Data by Detail ===
def filter_data_by_detail(metadata, tab, detail):
    """Filter metadata based on selected tab and detail"""
    filtered = {}
    
    # Define keywords for each detail type
    detail_keywords = {
        "Overall Performance Summary": ["performance", "summary", "total", "overall", "aggregate"],
        "Weekly Summary": ["week", "weekly", "w1", "w2", "w3", "w4"],
        "Daily Summary": ["day", "daily", "date", "d1", "d2", "d3"],
        "Overall Age and Gender": ["age", "gender", "male", "female", "demographic"],
        "Overall Hourly": ["hour", "hourly", "time", "h1", "h2", "morning", "evening"],
        "Network Summary": ["network", "channel", "platform", "media"],
        "Campaign Overview": ["campaign", "name", "title", "objective"],
        "Budget Allocation": ["budget", "cost", "spend", "allocation", "investment"],
        "Target Audience": ["target", "audience", "demographic", "segment"],
        "Performance Metrics": ["performance", "metrics", "kpi", "results"],
        "ROI Analysis": ["roi", "return", "profit", "revenue", "conversion"]
    }
    
    keywords = detail_keywords.get(detail, [])
    
    # Filter metadata based on keywords
    for key, value in metadata.items():
        key_lower = key.lower()
        if any(keyword in key_lower for keyword in keywords):
            filtered[key] = value
    
    # If no specific matches, try to find related fields
    if not filtered:
        for key, value in metadata.items():
            if any(word in key.lower() for word in detail.lower().split()):
                filtered[key] = value
    
    return filtered

# === Categorize Metadata ===
def categorize_metadata(metadata):
    """Categorize metadata fields for better organization"""
    categories = {
        "📊 Performance Metrics": {},
        "📅 Time & Date": {},
        "👥 Demographics": {},
        "💰 Financial": {},
        "🎯 Campaign Info": {},
        "📱 Technical": {},
        "📍 Other": {}
    }
    
    for key, value in metadata.items():
        key_lower = key.lower()
        
        if any(word in key_lower for word in ["performance", "metric", "kpi", "result", "score"]):
            categories["📊 Performance Metrics"][key] = value
        elif any(word in key_lower for word in ["date", "time", "hour", "day", "week", "month"]):
            categories["📅 Time & Date"][key] = value
        elif any(word in key_lower for word in ["age", "gender", "demographic", "audience"]):
            categories["👥 Demographics"][key] = value
        elif any(word in key_lower for word in ["cost", "budget", "spend", "revenue", "roi", "price"]):
            categories["💰 Financial"][key] = value
        elif any(word in key_lower for word in ["campaign", "name", "title", "objective", "goal"]):
            categories["🎯 Campaign Info"][key] = value
        elif any(word in key_lower for word in ["id", "index", "version", "type", "format"]):
            categories["📱 Technical"][key] = value
        else:
            categories["📍 Other"][key] = value
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

# === Sidebar Database Stats ===
if st.session_state.get('pinecone_connected', False):
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Database Info")
        
        if st.button("🔄 Refresh Stats"):
            try:
                pc, index, model = connect_to_pinecone(api_key, index_name)
                if index:
                    stats = index.describe_index_stats()
                    st.json(stats)
            except Exception as e:
                st.error(f"Error: {e}")

# === Run App ===
if __name__ == "__main__":
    main()
