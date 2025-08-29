from pinecone import Pinecone
import openai
from pinecone_text.sparse import BM25Encoder
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
import time
import json
import re

class FileBasedRAG:
    def __init__(self):
        # Pinecone setup
        self.pc = Pinecone(api_key="pcsk_6KzKjJ_JppwNYuBoYwRU2yBnUE3W9n8Cxk4xP2xvAeQeUSBG6gNKj1D3wMcKoXcAmg2FJW")
        self.index = self.pc.Index("campaign")
        
        # Initialize BM25 encoder
        self.encoder = BM25Encoder.default()
        
        # Report structure definition
        self.report_structure = [
            "Screen Details",
            "Overall Performance Summary", 
            "Daily Summary",
            "Overall Age and Gender",
            "Overall Hourly",
            "Network Summary"
        ]
    
    def search_by_identifier(self, identifier: str, search_type: str = "auto") -> List[Dict]:
        """
        Search by file name, network ID, or reference ID
        
        Args:
            identifier: The search term (filename, network_id, reference_id)
            search_type: "filename", "network_id", "reference_id", or "auto"
        """
        
        # Determine search type automatically if not specified
        if search_type == "auto":
            if "." in identifier and any(ext in identifier.lower() for ext in ['.csv', '.xlsx', '.json', '.txt']):
                search_type = "filename"
            elif identifier.isdigit() or re.match(r'^[A-Z0-9_-]+$', identifier):
                search_type = "network_id"
            else:
                search_type = "reference_id"
        
        # Create search filters based on type
        filter_dict = {}
        if search_type == "filename":
            filter_dict = {"filename": {"$eq": identifier}}
        elif search_type == "network_id":
            filter_dict = {"network_id": {"$eq": identifier}}
        elif search_type == "reference_id":
            filter_dict = {"reference_id": {"$eq": identifier}}
        
        # Search with sparse vector for better matching
        query_words = identifier.lower().split()
        sparse_vector = {
            "indices": list(range(min(10, len(query_words)))),
            "values": [0.8 - (i * 0.1) for i in range(min(10, len(query_words)))]
        }
        
        try:
            results = self.index.query(
                sparse_vector=sparse_vector,
                filter=filter_dict if filter_dict else None,
                top_k=20,
                include_metadata=True
            )
            return results.get('matches', [])
        except Exception as e:
            # Fallback to text search if filter fails
            results = self.index.query(
                sparse_vector=sparse_vector,
                top_k=20,
                include_metadata=True
            )
            
            # Filter results manually
            filtered_matches = []
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                if (search_type == "filename" and identifier.lower() in metadata.get('filename', '').lower()) or \
                   (search_type == "network_id" and identifier in str(metadata.get('network_id', ''))) or \
                   (search_type == "reference_id" and identifier.lower() in metadata.get('reference_id', '').lower()) or \
                   (search_type == "auto" and identifier.lower() in str(metadata).lower()):
                    filtered_matches.append(match)
            
            return filtered_matches
    
    def organize_data_by_tabs(self, matches: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize retrieved data into report structure tabs"""
        
        organized_data = {tab: [] for tab in self.report_structure}
        
        for match in matches:
            metadata = match.get('metadata', {})
            tab_type = metadata.get('report_section', 'Overall Performance Summary')
            
            # Map data to appropriate tabs based on content
            if 'screen' in str(metadata).lower() or 'device' in str(metadata).lower():
                organized_data['Screen Details'].append(metadata)
            elif 'daily' in str(metadata).lower() or 'date' in str(metadata).lower():
                organized_data['Daily Summary'].append(metadata)
            elif any(term in str(metadata).lower() for term in ['age', 'gender', 'demographic']):
                organized_data['Overall Age and Gender'].append(metadata)
            elif 'hour' in str(metadata).lower() or 'time' in str(metadata).lower():
                organized_data['Overall Hourly'].append(metadata)
            elif 'network' in str(metadata).lower():
                organized_data['Network Summary'].append(metadata)
            else:
                organized_data['Overall Performance Summary'].append(metadata)
        
        return organized_data
    
    def create_tab_visualizations(self, tab_data: Dict[str, List[Dict]], tab_name: str) -> List[go.Figure]:
        """Create visualizations specific to each tab"""
        
        if not tab_data or len(tab_data) == 0:
            return []
        
        charts = []
        df = pd.DataFrame(tab_data)
        
        if tab_name == "Screen Details":
            # Screen/Device performance charts
            if 'device_type' in df.columns and 'impressions' in df.columns:
                fig = px.pie(df, values='impressions', names='device_type', 
                           title='📱 Impressions by Device Type')
                charts.append(fig)
            
            if 'screen_size' in df.columns and 'ctr' in df.columns:
                fig = px.bar(df, x='screen_size', y='ctr', 
                           title='📺 CTR by Screen Size')
                charts.append(fig)
        
        elif tab_name == "Overall Performance Summary":
            # General performance metrics
            if 'impressions' in df.columns and 'clicks' in df.columns:
                fig = px.scatter(df, x='impressions', y='clicks', 
                               title='👁️ Impressions vs Clicks Performance')
                charts.append(fig)
            
            if 'spend' in df.columns and 'revenue' in df.columns:
                fig = px.bar(df, x=df.index, y=['spend', 'revenue'], 
                           title='💰 Spend vs Revenue Comparison')
                charts.append(fig)
        
        elif tab_name == "Daily Summary":
            # Time-based analysis
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if 'impressions' in df.columns:
                    fig = px.line(df, x='date', y='impressions', 
                                title='📈 Daily Impressions Trend')
                    charts.append(fig)
                
                if 'ctr' in df.columns:
                    fig = px.line(df, x='date', y='ctr', 
                                title='📊 Daily CTR Trend')
                    charts.append(fig)
        
        elif tab_name == "Overall Age and Gender":
            # Demographic analysis
            if 'age_group' in df.columns and 'impressions' in df.columns:
                fig = px.bar(df, x='age_group', y='impressions', 
                           title='👥 Impressions by Age Group')
                charts.append(fig)
            
            if 'gender' in df.columns and 'ctr' in df.columns:
                fig = px.box(df, x='gender', y='ctr', 
                           title='⚧️ CTR Distribution by Gender')
                charts.append(fig)
        
        elif tab_name == "Overall Hourly":
            # Hourly performance
            if 'hour' in df.columns and 'impressions' in df.columns:
                fig = px.line(df, x='hour', y='impressions', 
                            title='🕐 Hourly Impressions Pattern')
                charts.append(fig)
            
            if 'hour' in df.columns and 'ctr' in df.columns:
                fig = px.heatmap(df.pivot_table(values='ctr', index='hour', aggfunc='mean'), 
                               title='🔥 Hourly CTR Heatmap')
                charts.append(fig)
        
        elif tab_name == "Network Summary":
            # Network performance
            if 'network_name' in df.columns and 'revenue' in df.columns:
                fig = px.treemap(df, path=['network_name'], values='revenue',
                               title='🌐 Revenue by Network')
                charts.append(fig)
            
            if 'network_id' in df.columns and 'cpm' in df.columns:
                fig = px.bar(df, x='network_id', y='cpm', 
                           title='💳 CPM by Network ID')
                charts.append(fig)
        
        return charts
    
    def generate_tab_summary(self, tab_data: List[Dict], tab_name: str) -> str:
        """Generate summary for each tab"""
        
        if not tab_data:
            return f"No data available for {tab_name}"
        
        df = pd.DataFrame(tab_data)
        summary = f"## 📊 {tab_name} Summary\n\n"
        
        # General metrics
        summary += f"**Total Records:** {len(df)}\n\n"
        
        # Tab-specific summaries
        if tab_name == "Screen Details":
            if 'device_type' in df.columns:
                device_counts = df['device_type'].value_counts()
                summary += f"**Top Device Type:** {device_counts.index[0]} ({device_counts.iloc[0]} records)\n"
        
        elif tab_name == "Overall Performance Summary":
            if 'impressions' in df.columns:
                total_impressions = df['impressions'].sum()
                summary += f"**Total Impressions:** {total_impressions:,}\n"
            if 'clicks' in df.columns:
                total_clicks = df['clicks'].sum()
                summary += f"**Total Clicks:** {total_clicks:,}\n"
            if 'spend' in df.columns:
                total_spend = df['spend'].sum()
                summary += f"**Total Spend:** ${total_spend:,.2f}\n"
        
        elif tab_name == "Daily Summary":
            if 'date' in df.columns:
                date_range = f"{df['date'].min()} to {df['date'].max()}"
                summary += f"**Date Range:** {date_range}\n"
        
        elif tab_name == "Network Summary":
            if 'network_name' in df.columns:
                unique_networks = df['network_name'].nunique()
                summary += f"**Unique Networks:** {unique_networks}\n"
        
        # Add key metrics if available
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            summary += "\n**Key Metrics:**\n"
            for col in numeric_columns[:5]:  # Show top 5 numeric columns
                if col in df.columns and not df[col].isna().all():
                    avg_val = df[col].mean()
                    summary += f"- **Average {col.title()}:** {avg_val:.2f}\n"
        
        return summary


def create_file_interface():
    """Create Streamlit interface for file-based search"""
    
    st.set_page_config(
        page_title="📁 File-Based RAG System", 
        page_icon="🔍",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .search-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .tab-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📁 File-Based RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### *Search by File Name, Network ID, or Reference ID*")
    
    # Initialize RAG system
    if 'file_rag' not in st.session_state:
        st.session_state.file_rag = FileBasedRAG()
    
    # Search interface
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Enter File Name, Network ID, or Reference ID:",
            placeholder="e.g., campaign_data.csv, NETWORK_123, REF_456"
        )
    
    with col2:
        search_type = st.selectbox(
            "Search Type:",
            ["auto", "filename", "network_id", "reference_id"]
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Example searches
    st.markdown("### 💡 Example Searches")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("📄 campaign_report.csv"):
            st.session_state.example_query = "campaign_report.csv"
            st.session_state.example_type = "filename"
    
    with example_col2:
        if st.button("🌐 NETWORK_12345"):
            st.session_state.example_query = "NETWORK_12345"
            st.session_state.example_type = "network_id"
    
    with example_col3:
        if st.button("🔗 REF_ABC123"):
            st.session_state.example_query = "REF_ABC123"
            st.session_state.example_type = "reference_id"
    
    # Use example if clicked
    if hasattr(st.session_state, 'example_query'):
        search_query = st.session_state.example_query
        search_type = st.session_state.example_type
        search_button = True
        # Clear example
        delattr(st.session_state, 'example_query')
        delattr(st.session_state, 'example_type')
    
    # Process search
    if search_button and search_query:
        with st.spinner(f"🔍 Searching for {search_query}..."):
            try:
                # Search for matches
                matches = st.session_state.file_rag.search_by_identifier(search_query, search_type)
                
                if matches:
                    st.success(f"✅ Found {len(matches)} results for '{search_query}'")
                    
                    # Organize data by tabs
                    organized_data = st.session_state.file_rag.organize_data_by_tabs(matches)
                    
                    # Display results in tabs
                    tab_objects = st.tabs([f"📊 {tab}" for tab in st.session_state.file_rag.report_structure])
                    
                    for idx, (tab_name, tab_data) in enumerate(organized_data.items()):
                        with tab_objects[idx]:
                            if tab_data:
                                # Tab summary
                                summary = st.session_state.file_rag.generate_tab_summary(tab_data, tab_name)
                                st.markdown(summary)
                                
                                # Create visualizations
                                charts = st.session_state.file_rag.create_tab_visualizations(tab_data, tab_name)
                                
                                if charts:
                                    st.subheader("📈 Visualizations")
                                    for chart in charts:
                                        st.plotly_chart(chart, use_container_width=True)
                                
                                # Data table
                                st.subheader("📋 Detailed Data")
                                df = pd.DataFrame(tab_data)
                                if not df.empty:
                                    # Show key columns first
                                    key_columns = ['filename', 'network_id', 'reference_id', 'date', 'impressions', 'clicks', 'spend', 'revenue']
                                    available_key_columns = [col for col in key_columns if col in df.columns]
                                    remaining_columns = [col for col in df.columns if col not in available_key_columns]
                                    
                                    display_columns = available_key_columns + remaining_columns
                                    st.dataframe(df[display_columns], use_container_width=True)
                                else:
                                    st.info(f"No data available in {tab_name}")
                            else:
                                st.info(f"No data found for {tab_name}")
                    
                    # Overall metrics
                    st.markdown("---")
                    st.subheader("📊 Overall Metrics")
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    all_data = []
                    for tab_data in organized_data.values():
                        all_data.extend(tab_data)
                    
                    if all_data:
                        df_all = pd.DataFrame(all_data)
                        
                        with metric_col1:
                            st.markdown(f"""
                            <div class="metric-box">
                                <h3>{len(matches)}</h3>
                                <p>Total Records</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col2:
                            unique_files = df_all['filename'].nunique() if 'filename' in df_all.columns else 0
                            st.markdown(f"""
                            <div class="metric-box">
                                <h3>{unique_files}</h3>
                                <p>Unique Files</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col3:
                            unique_networks = df_all['network_id'].nunique() if 'network_id' in df_all.columns else 0
                            st.markdown(f"""
                            <div class="metric-box">
                                <h3>{unique_networks}</h3>
                                <p>Networks</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col4:
                            total_impressions = df_all['impressions'].sum() if 'impressions' in df_all.columns else 0
                            st.markdown(f"""
                            <div class="metric-box">
                                <h3>{total_impressions:,.0f}</h3>
                                <p>Total Impressions</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                else:
                    st.warning(f"❌ No results found for '{search_query}' with search type '{search_type}'")
                    st.info("💡 Try different search terms or change the search type")
                    
            except Exception as e:
                st.error(f"❌ Error occurred during search: {str(e)}")
                st.info("🔧 Please check your search query and try again")
    
    # Sidebar with information
    st.sidebar.markdown("### 📋 Report Structure")
    for i, tab in enumerate(st.session_state.file_rag.report_structure, 1):
        st.sidebar.markdown(f"{i}. **{tab}**")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Search Types")
    st.sidebar.markdown("""
    - **Auto**: Automatically detect search type
    - **Filename**: Search by file name (e.g., report.csv)  
    - **Network ID**: Search by network identifier
    - **Reference ID**: Search by reference identifier
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - Use exact file names for best results
    - Network IDs are usually alphanumeric
    - Reference IDs can contain letters and numbers
    - Try different search types if no results found
    """)
    
    # Database status
    st.sidebar.markdown("---")
    st.sidebar.header("🗄️ Database Status")
    try:
        stats = st.session_state.file_rag.index.describe_index_stats()
        st.sidebar.metric("Total Records", f"{stats.total_vector_count:,}")
        st.sidebar.metric("Database Fullness", f"{stats.index_fullness:.1%}")
        
        if stats.index_fullness > 0.8:
            st.sidebar.success("✅ Database is well populated!")
        elif stats.index_fullness > 0.5:
            st.sidebar.info("ℹ️ Database is moderately populated")
        else:
            st.sidebar.warning("⚠️ Database needs more data")
            
    except Exception as e:
        st.sidebar.error("❌ Could not connect to database")

if __name__ == "__main__":
    create_file_interface()
