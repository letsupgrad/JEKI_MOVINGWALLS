from pinecone import Pinecone
import openai # Note: openai is imported but not used, can be removed if not needed for future features.
from pinecone_text.sparse import BM25Encoder
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import re

class JekiDataRAG:
    def __init__(self):
        # --- API KEY IS HARDCODED HERE ---
        # DANGER: This is not recommended for production or shared code.
        # Replace with your actual Pinecone API key.
        PINECONE_API_KEY = "pcsk_3wbxiS_JFsW8uFyumkQ2oMD5FkfjKJPV5kYkiDwX1T15tg2HtFSn4ioZEeVpsSV6V1DK7s"
        
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.index = self.pc.Index("campaign")
        except Exception as e:
            st.error(f"Failed to connect to Pinecone. Please check your API key and network connection. Error: {e}")
            self.pc = None
            self.index = None
        
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
        Search by file name, network ID, or reference ID in Pinecone.
        """
        if not self.index:
            st.error("Pinecone index is not available.")
            return []
            
        # Auto-detect search type
        if search_type == "auto":
            if "." in identifier and any(ext in identifier.lower() for ext in ['.csv', '.xlsx', '.json', '.txt', '.pdf']):
                search_type = "filename"
            elif identifier.startswith("JPN-JEK") or re.match(r'^[A-Z]+-[A-Z]+-[A-Z]-\d+-\d+$', identifier):
                search_type = "reference_id"
            elif identifier.isdigit() or re.match(r'^[A-Z0-9_-]+$', identifier):
                search_type = "network_id"
            else:
                search_type = "filename"
        
        # Create a simple sparse vector for keyword matching
        query_text = identifier.lower().replace("-", " ").replace("_", " ")
        query_words = query_text.split()
        sparse_vector = {
            "indices": list(range(len(query_words))),
            "values": [1.0] * len(query_words) # Simple 1.0 weight for each word
        }
        
        try:
            # Query Pinecone. top_k is increased to allow for more comprehensive local filtering.
            results = self.index.query(
                sparse_vector=sparse_vector,
                top_k=100,
                include_metadata=True
            )
            
            # Post-filter results for higher accuracy
            filtered_matches = []
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                match_found = False
                
                if search_type == "filename":
                    if identifier.lower() in metadata.get('filename', '').lower():
                        match_found = True
                elif search_type == "reference_id":
                    if identifier == metadata.get('reference_id', ''):
                        match_found = True
                elif search_type == "network_id":
                    if identifier == str(metadata.get('network_id', '')):
                        match_found = True
                else: # General fallback
                    if identifier.lower() in str(metadata).lower():
                        match_found = True
                
                if match_found:
                    filtered_matches.append(match)
            
            return filtered_matches
            
        except Exception as e:
            st.error(f"Error during Pinecone search: {str(e)}")
            return []
    
    def organize_data_by_tabs(self, matches: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize retrieved metadata into the 6 report structure tabs."""
        organized_data = {tab: [] for tab in self.report_structure}
        
        for match in matches:
            metadata = match.get('metadata', {})
            
            # Use specific keys for more accurate classification
            if any(key in metadata for key in ['reference_id', 'display_name', 'screen_name']):
                organized_data['Screen Details'].append(metadata)
            elif any(key in metadata for key in ['age_group', 'gender']):
                organized_data['Overall Age and Gender'].append(metadata)
            elif 'date' in metadata:
                organized_data['Daily Summary'].append(metadata)
            elif 'hour' in metadata:
                organized_data['Overall Hourly'].append(metadata)
            elif any(key in metadata for key in ['network_id', 'campaign_name']):
                 organized_data['Network Summary'].append(metadata)
            elif any(key in metadata for key in ['impressions', 'reach', 'frequency']):
                organized_data['Overall Performance Summary'].append(metadata)
            else:
                # Default bucket if no specific keys match
                organized_data['Overall Performance Summary'].append(metadata)
        
        return organized_data
    
    def create_tab_visualizations(self, tab_data: List[Dict], tab_name: str) -> List[go.Figure]:
        """Create visualizations for each tab."""
        if not tab_data:
            return []
        
        charts = []
        df = pd.DataFrame(tab_data).dropna(how='all', axis=1)
        
        if tab_name == "Screen Details":
            if 'reference_id' in df.columns and 'impressions' in df.columns:
                df_sorted = df.sort_values('impressions', ascending=False).head(15)
                fig = px.bar(df_sorted, x='reference_id', y='impressions', 
                           title='Top 15 Screens by Impressions',
                           labels={'impressions': 'Total Impressions', 'reference_id': 'Reference ID'})
                fig.update_xaxes(tickangle=45, type='category')
                charts.append(fig)
            if 'display_name' in df.columns and 'impressions' in df.columns:
                df_grouped = df.groupby('display_name')['impressions'].sum().nlargest(10).reset_index()
                fig = px.pie(df_grouped, values='impressions', names='display_name', title='Top 10 Stations by Impressions')
                charts.append(fig)
        
        elif tab_name == "Overall Performance Summary":
            if 'impressions' in df.columns and 'display_name' in df.columns:
                top_locations = df.groupby('display_name')['impressions'].sum().nlargest(10).reset_index()
                fig = px.bar(top_locations, x='impressions', y='display_name', orientation='h', title='Top 10 Locations by Impressions')
                charts.append(fig)
            if all(c in df.columns for c in ['impressions', 'reach', 'frequency', 'display_name']):
                df_sample = df.dropna(subset=['impressions', 'reach', 'frequency']).head(20)
                fig = px.scatter(df_sample, x='reach', y='frequency', size='impressions', color='display_name',
                               hover_name='display_name', title='Reach vs. Frequency Analysis')
                charts.append(fig)

        elif tab_name == "Daily Summary":
            if 'date' in df.columns and 'impressions' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                daily_data = df.groupby('date')['impressions'].sum().reset_index().dropna()
                if not daily_data.empty:
                    fig = px.line(daily_data, x='date', y='impressions', title='Daily Impressions Trend', markers=True)
                    charts.append(fig)
        
        elif tab_name == "Overall Age and Gender":
            if 'age_group' in df.columns and 'impressions' in df.columns:
                age_data = df.groupby('age_group')['impressions'].sum().reset_index()
                fig = px.bar(age_data, x='age_group', y='impressions', title='Impressions by Age Group')
                fig.update_xaxes(type='category')
                charts.append(fig)
            if 'gender' in df.columns and 'impressions' in df.columns:
                gender_data = df.groupby('gender')['impressions'].sum().reset_index()
                fig = px.pie(gender_data, values='impressions', names='gender', title='Impressions by Gender')
                charts.append(fig)
        
        elif tab_name == "Overall Hourly":
            if 'hour' in df.columns and 'impressions' in df.columns:
                df['hour_val'] = pd.to_numeric(df['hour'], errors='coerce')
                hourly_data = df.dropna(subset=['hour_val']).groupby('hour_val')['impressions'].sum().reset_index()
                if not hourly_data.empty:
                    fig = px.line(hourly_data, x='hour_val', y='impressions', title='Hourly Impressions Pattern', markers=True)
                    fig.update_xaxes(title_text='Hour of Day (24h)')
                    charts.append(fig)
        
        elif tab_name == "Network Summary":
            if 'network_id' in df.columns and 'impressions' in df.columns:
                network_data = df.groupby('network_id')['impressions'].sum().reset_index()
                fig = px.treemap(network_data, path=['network_id'], values='impressions', title='Network Performance Treemap')
                charts.append(fig)
        
        return charts
    
    def generate_tab_summary(self, tab_data: List[Dict], tab_name: str) -> str:
        """Generate a markdown summary for each tab."""
        if not tab_data:
            return f"**{tab_name}**: No data available for this category."
        
        df = pd.DataFrame(tab_data)
        summary = f"### {tab_name} Summary\n"
        summary += f"- **Total Records Analyzed**: {len(df)}\n"
        
        if tab_name == "Screen Details" and 'impressions' in df.columns and 'display_name' in df.columns:
            if 'reference_id' in df.columns: summary += f"- **Unique Screens**: {df['reference_id'].nunique()}\n"
            if 'display_name' in df.columns: summary += f"- **Unique Locations**: {df['display_name'].nunique()}\n"
            top_loc = df.loc[df['impressions'].idxmax()]
            summary += f"- **Top Location**: {top_loc['display_name']} ({int(top_loc['impressions']):,} impressions)\n"
        
        elif tab_name == "Overall Performance Summary":
            if 'impressions' in df.columns: summary += f"- **Total Impressions**: {int(df['impressions'].sum()):,}\n"
            if 'reach' in df.columns: summary += f"- **Total Reach**: {int(df['reach'].sum()):,}\n"
            if 'frequency' in df.columns: summary += f"- **Average Frequency**: {df['frequency'].mean():.2f}\n"

        elif tab_name == "Daily Summary" and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dropna()
            if not df.empty:
                date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
                summary += f"- **Date Range**: {date_range}\n"
                if 'impressions' in df.columns:
                    daily_avg = df.groupby(df['date'].dt.date)['impressions'].sum().mean()
                    summary += f"- **Average Daily Impressions**: {int(daily_avg):,}\n"
        
        elif tab_name == "Overall Age and Gender" and 'impressions' in df.columns:
            if 'age_group' in df.columns: summary += f"- **Top Age Group**: {df.groupby('age_group')['impressions'].sum().idxmax()}\n"
            if 'gender' in df.columns: summary += f"- **Gender Distribution**: {df.groupby('gender')['impressions'].sum().to_dict()}\n"
        
        elif tab_name == "Overall Hourly" and 'hour' in df.columns and 'impressions' in df.columns:
            df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
            peak_df = df.groupby('hour')['impressions'].sum()
            if not peak_df.empty:
                peak_hour = int(peak_df.idxmax())
                summary += f"- **Peak Hour**: {peak_hour}:00 - {peak_hour+1}:00\n"

        elif tab_name == "Network Summary":
            if 'network_id' in df.columns: summary += f"- **Unique Networks**: {df['network_id'].nunique()}\n"
            if 'campaign_name' in df.columns: summary += f"- **Unique Campaigns**: {df['campaign_name'].nunique()}\n"
        return summary

def create_jeki_interface():
    """Create the full Streamlit interface."""
    st.set_page_config(page_title="JEKI Data RAG", page_icon="🚋", layout="wide")
    
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem;
                   background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">JEKI Data Integration - Pinecone RAG</h1>', unsafe_allow_html=True)
    st.markdown("### Search Train Channel & JAD Vision Campaign Data by File Name, Network ID, or Reference ID")
    
    if 'jeki_rag' not in st.session_state:
        st.session_state.jeki_rag = JekiDataRAG()
    
    # Search interface
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            search_query = st.text_input("Search Query:", placeholder="e.g., JAD_Vision_Report.pdf, JPN-JEK-D-00000-00030", label_visibility="collapsed")
        with col2:
            search_type = st.selectbox("Search Type:", ["auto", "filename", "network_id", "reference_id"], label_visibility="collapsed")
        with col3:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    if search_button and search_query:
        with st.spinner(f"Searching for '{search_query}'..."):
            matches = st.session_state.jeki_rag.search_by_identifier(search_query, search_type)
            
            if matches:
                st.success(f"Found {len(matches)} relevant records for '{search_query}'.")
                organized_data = st.session_state.jeki_rag.organize_data_by_tabs(matches)
                
                tab_objects = st.tabs([f"📊 {tab}" for tab in st.session_state.jeki_rag.report_structure])
                
                for idx, (tab_name, tab_data) in enumerate(organized_data.items()):
                    with tab_objects[idx]:
                        if tab_data:
                            summary = st.session_state.jeki_rag.generate_tab_summary(tab_data, tab_name)
                            st.markdown(summary)
                            
                            charts = st.session_state.jeki_rag.create_tab_visualizations(tab_data, tab_name)
                            if charts:
                                st.subheader("📈 Visualizations")
                                for chart in charts:
                                    st.plotly_chart(chart, use_container_width=True)
                            
                            st.subheader("📋 Detailed Data")
                            df = pd.DataFrame(tab_data)
                            priority_cols = ['reference_id', 'display_name', 'impressions', 'reach', 'frequency', 'date', 'hour', 'age_group', 'gender', 'filename']
                            display_cols = [c for c in priority_cols if c in df.columns] + [c for c in df.columns if c not in priority_cols]
                            st.dataframe(df[display_cols], use_container_width=True, height=300)
                            
                            csv = df[display_cols].to_csv(index=False).encode('utf-8')
                            st.download_button(label=f"📥 Download Data", data=csv, file_name=f"{tab_name.replace(' ', '_').lower()}_data.csv", mime="text/csv")
                        else:
                            st.info(f"No data classified under '{tab_name}' for this search.")
            else:
                st.warning(f"No results found for '{search_query}'. Please try a different term.")
    
    # Sidebar Information
    st.sidebar.header("📋 Report Structure")
    st.sidebar.markdown("\n".join(f"- **{tab}**" for tab in st.session_state.jeki_rag.report_structure))
    
    st.sidebar.divider()
    st.sidebar.header("🔍 Example Searches")
    st.sidebar.markdown("""
    - `JAD Vision` (Filename)
    - `Train Channel` (Filename)
    - `JPN-JEK-D-00000-00030` (Reference ID)
    - `JPN-JEK` (Network ID)
    """)
    
    st.sidebar.divider()
    st.sidebar.header("🗄️ Database Status")
    if st.session_state.jeki_rag.index:
        try:    
            stats = st.session_state.jeki_rag.index.describe_index_stats()
            st.sidebar.metric("Total Vectors", f"{stats.total_vector_count:,}")
            st.sidebar.metric("Index Fullness", f"{stats.index_fullness:.2%}")
            st.sidebar.success("🟢 Pinecone Connected")
        except Exception as e:
            st.sidebar.error("🔴 Pinecone Connection Failed")
            st.sidebar.caption(str(e))
    else:
        st.sidebar.error("🔴 Pinecone Not Connected")

if __name__ == "__main__":
    create_jeki_interface()
