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
    ],
    "Report_Info_Tab": [
        "Screen Details",
        "Worksheet Information",
        "Column Descriptions",
        "Field Explanations",
        "Data Structure"
    ],
    "Japanese_Content_Tab": [
        "ワークシート名 (Worksheet Name)",
        "画面詳細 (Screen Details)", 
        "欄説明 (Column Explanation)",
        "データ構造 (Data Structure)",
        "レポート情報 (Report Info)"
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
        # Smart default searches based on what we see in the data
        default_searches = {
            "Network Summary": "Network Summary",
            "Screen Details": "Screen Details",
            "Campaign Overview": "Dell campaign",
            "Overall Performance Summary": "performance summary",
            "Weekly Summary": "weekly",
            "Daily Summary": "daily"
        }
        
        default_search = default_searches.get(selected_detail, "Dell")
        
        search_query = st.text_input(
            "Enter search terms:",
            placeholder="e.g., Dell campaign, Screen Details, Network Summary...",
            value=default_search
        )
    
    with col2:
        num_results = st.number_input("Results", min_value=1, max_value=50, value=15)
    
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
                    
                    # Show database stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records in DB", "13,509")
                    with col2:
                        st.metric("Search Results", len(results['matches']))
                    with col3:
                        best_score = max([match['score'] for match in results['matches']])
                        st.metric("Best Match Score", f"{best_score:.4f}")
                    
                    st.markdown("---")
                    
                    # Show preview of all results first
                    st.markdown("### 📋 Search Results Preview")
                    preview_data = []
                    
                    for i, match in enumerate(results['matches']):
                        metadata = match['metadata']
                        # Get meaningful fields for preview, prioritizing certain fields
                        meaningful_fields = {k: v for k, v in metadata.items() if v and str(v).strip() != ""}
                        
                        # Create better preview focusing on key fields
                        preview_parts = []
                        priority_fields = ['Screen Details', 'ワークシート名', '画面詳細', 'Report Name']
                        
                        for field in priority_fields:
                            if field in meaningful_fields and meaningful_fields[field]:
                                preview_parts.append(f"{field}: {meaningful_fields[field][:50]}...")
                        
                        if not preview_parts:
                            # Fallback to any meaningful field
                            for k, v in list(meaningful_fields.items())[:2]:
                                if len(str(v)) > 3:
                                    preview_parts.append(f"{k}: {str(v)[:50]}...")
                        
                        preview_text = " | ".join(preview_parts) if preview_parts else "No meaningful preview data"
                        
                        preview_data.append({
                            "Rank": i + 1,
                            "Record ID": match['id'],
                            "Score": f"{match['score']:.4f}",
                            "Preview": preview_text,
                            "Fields": len(meaningful_fields)
                        })
                    
                    # Display preview table
                    preview_df = pd.DataFrame(preview_data)
                    st.dataframe(preview_df, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Detailed Record Views")
                    
                    # Add filter options
                    col1, col2 = st.columns(2)
                    with col1:
                        min_score = st.slider("Minimum Similarity Score", 0.0, 1.0, 0.0, 0.01)
                    with col2:
                        show_empty = st.checkbox("Show records with empty fields", value=True)
                    
                    # Filter results based on user preferences
                    filtered_matches = [
                        match for match in results['matches'] 
                        if match['score'] >= min_score
                    ]
                    
                    if not show_empty:
                        filtered_matches = [
                            match for match in filtered_matches
                            if any(v and str(v).strip() for v in match['metadata'].values())
                        ]
                    
                    # Display detailed results
                    for i, match in enumerate(filtered_matches):
                        meaningful_fields = {k: v for k, v in match['metadata'].items() if v and str(v).strip() != ""}
                        
                        # Create better expander title
                        if meaningful_fields:
                            # Try to get the most relevant field for title
                            title_field = None
                            for field in ['Screen Details', 'ワークシート名', '画面詳細', 'Report Name']:
                                if field in meaningful_fields:
                                    title_field = f"{field}: {str(meaningful_fields[field])[:100]}"
                                    break
                            
                            if not title_field:
                                first_meaningful = list(meaningful_fields.items())[0]
                                title_field = f"{first_meaningful[0]}: {str(first_meaningful[1])[:100]}"
                            
                            title = f"📊 Record #{match['id']} | {title_field} | Score: {match['score']:.4f}"
                        else:
                            title = f"📊 Record #{match['id']} - Empty/Header Row | Score: {match['score']:.4f}"
                        
                        with st.expander(title, expanded=(i < 3)):  # Auto-expand first 3
                            display_record_details(match['metadata'], selected_tab, selected_detail, match['id'])
                else:
                    st.warning("❌ No matching records found")
                    st.markdown("**Try these suggestions:**")
                    st.markdown("- Use terms like 'Screen Details', 'Network Summary', or 'Dell'")
                    st.markdown("- Try Japanese terms like 'ワークシート' or '画面詳細'")
                    st.markdown("- Use broader search terms")
                    st.markdown("- Check if your search terms match the content in your data")
                    
            except Exception as e:
                st.error(f"❌ Search Error: {e}")
                st.markdown("**Debug info:**")
                st.markdown(f"- Query: {search_query}")
                st.markdown(f"- Selected detail: {selected_detail}")
                st.markdown(f"- Number of results requested: {num_results}")

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
    
    # Header with better styling
    st.markdown(f"### 📋 Record Details")
    
    # Show record info in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Record ID", record_id)
    with col2:
        st.metric("Tab Category", selected_tab.replace("_", " ").title())
    with col3:
        st.metric("Detail Type", selected_detail)
    
    st.markdown("---")
    
    # Check if metadata has meaningful data
    meaningful_data = {k: v for k, v in metadata.items() if v and str(v).strip() != ""}
    empty_data = {k: v for k, v in metadata.items() if not v or str(v).strip() == ""}
    
    # Create tabs for organized display
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Filtered View", "📊 Available Data", "❌ Empty Fields", "🔧 Raw JSON"])
    
    with tab1:
        st.markdown(f"#### 🎯 {selected_detail} Specific Data")
        
        # Filter data based on selected detail
        filtered_data = filter_data_by_detail(meaningful_data, selected_tab, selected_detail)
        
        if filtered_data:
            # Display in a nice card format
            for key, value in filtered_data.items():
                st.info(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.warning(f"⚠️ No specific data found for '{selected_detail}' in this record")
            st.markdown("**Suggestions:**")
            st.markdown("- Try a different detail type from the dropdown")
            st.markdown("- Check the 'Available Data' tab to see what information exists")
            st.markdown("- This record might be a header or metadata row")
    
    with tab2:
        st.markdown("#### 📊 Available Data Fields")
        
        if meaningful_data:
            # Group and display meaningful data
            categorized_data = categorize_metadata(meaningful_data)
            
            for category, fields in categorized_data.items():
                if fields:
                    with st.expander(f"{category} ({len(fields)} fields)"):
                        for key, value in fields.items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.error("❌ No meaningful data found in this record")
            st.markdown("**This might be:**")
            st.markdown("- A header row from the Excel file")
            st.markdown("- A metadata or configuration row")
            st.markdown("- An empty/placeholder record")
    
    with tab3:
        st.markdown("#### ❌ Empty or Missing Fields")
        
        if empty_data:
            st.warning(f"Found {len(empty_data)} empty fields in this record:")
            
            # Show empty fields in columns
            empty_fields = list(empty_data.keys())
            if empty_fields:
                # Split into chunks of 3 for display
                for i in range(0, len(empty_fields), 3):
                    cols = st.columns(3)
                    for j, field in enumerate(empty_fields[i:i+3]):
                        with cols[j]:
                            st.write(f"• {field.replace('_', ' ').title()}")
        else:
            st.success("✅ All fields have data!")
    
    with tab4:
        st.markdown("#### 🔧 Raw Data (JSON)")
        
        # Show JSON with better formatting
        if metadata:
            st.json(metadata)
        else:
            st.error("No metadata available")

# === Filter Data by Detail ===
def filter_data_by_detail(metadata, tab, detail):
    """Filter metadata based on selected tab and detail - Enhanced for multilingual content"""
    filtered = {}
    
    # Enhanced keywords including Japanese terms and actual field names from the data
    detail_keywords = {
        "Overall Performance Summary": ["performance", "summary", "total", "overall", "aggregate", "result", "outcome"],
        "Weekly Summary": ["week", "weekly", "w1", "w2", "w3", "w4", "seven", "7"],
        "Daily Summary": ["day", "daily", "date", "d1", "d2", "d3", "today", "yesterday"],
        "Overall Age and Gender": ["age", "gender", "male", "female", "demographic", "men", "women", "m", "f"],
        "Overall Hourly": ["hour", "hourly", "time", "h1", "h2", "morning", "evening", "am", "pm"],
        "Network Summary": ["network", "channel", "platform", "media", "tv", "radio", "digital", "Network Summary"],
        "Campaign Overview": ["campaign", "name", "title", "objective", "dell", "vision", "jad", "Campaign Overview"],
        "Budget Allocation": ["budget", "cost", "spend", "allocation", "investment", "money", "dollar"],
        "Target Audience": ["target", "audience", "demographic", "segment", "group"],
        "Performance Metrics": ["performance", "metrics", "kpi", "results", "clicks", "views"],
        "ROI Analysis": ["roi", "return", "profit", "revenue", "conversion", "ctr", "cpc"],
        
        # New categories based on actual data
        "Screen Details": ["Screen Details", "screen", "details", "画面詳細", "画面"],
        "Worksheet Information": ["ワークシート名", "worksheet", "sheet", "tab"],
        "Column Descriptions": ["欄説明", "欄", "column", "field", "description"],
        "Field Explanations": ["explanation", "説明", "definition", "meaning"],
        "Data Structure": ["structure", "format", "layout", "構造"],
        
        # Japanese content categories
        "ワークシート名 (Worksheet Name)": ["ワークシート名", "worksheet", "sheet name"],
        "画面詳細 (Screen Details)": ["画面詳細", "Screen Details", "screen", "画面"],
        "欄説明 (Column Explanation)": ["欄説明", "欄", "column", "explanation"],
        "データ構造 (Data Structure)": ["データ構造", "structure", "構造", "data"],
        "レポート情報 (Report Info)": ["レポート情報", "Report Info", "report", "information"]
    }
    
    keywords = detail_keywords.get(detail, [])
    
    # First pass: Exact field name matching (for fields like "Screen Details")
    for key, value in metadata.items():
        if value and str(value).strip():
            if key in keywords:  # Exact match
                filtered[key] = value
            elif any(keyword.lower() == key.lower() for keyword in keywords):  # Case insensitive exact match
                filtered[key] = value
    
    # Second pass: Keyword matching in field names
    if not filtered:
        for key, value in metadata.items():
            if value and str(value).strip():
                key_lower = key.lower()
                if any(keyword.lower() in key_lower for keyword in keywords):
                    filtered[key] = value
    
    # Third pass: Content matching (search within the values)
    if not filtered:
        for key, value in metadata.items():
            if value and str(value).strip():
                value_lower = str(value).lower()
                if any(keyword.lower() in value_lower for keyword in keywords):
                    filtered[key] = value
    
    # Fourth pass: Broader matching for specific cases
    if not filtered:
        detail_words = detail.lower().replace("_", " ").replace("(", "").replace(")", "").split()
        for key, value in metadata.items():
            if value and str(value).strip():
                key_lower = key.lower()
                value_lower = str(value).lower()
                if any(word in key_lower or word in value_lower for word in detail_words):
                    filtered[key] = value
    
    # Special case: If looking for Network Summary and we have the exact content
    if "network summary" in detail.lower() and not filtered:
        for key, value in metadata.items():
            if value and "network summary" in str(value).lower():
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
