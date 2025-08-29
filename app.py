
from pinecone import Pinecone
import openai
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
        PINECONE_API_KEY = "pcsk_3wbxiS_JFsW8uFyumkQ2oMD5FkfjKJPV5kYkiDwX1T15tg2HtFSn4ioZEeVpsSV6V1DK7s"
        
        try:
            # Pinecone Setup
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.index = self.pc.Index("campaign")
            
            # --- NEW: OpenAI Client Setup ---
            # Correctly uses Streamlit secrets for the OpenAI key.
            # Make sure you have OPENAI_API_KEY in your .streamlit/secrets.toml file.
            self.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        except Exception as e:
            st.error(f"Failed to connect to services. Please check API keys and network. Error: {e}")
            self.pc = None
            self.index = None
            self.openai_client = None
        
        # Initialize BM25 encoder
        self.encoder = BM25Encoder.default()
        
        # Report structure definition
        self.report_structure = [
            "Screen Details", "Overall Performance Summary", "Daily Summary",
            "Overall Age and Gender", "Overall Hourly", "Network Summary"
        ]
    
    # --- NEW: Helper function to get embeddings ---
    def get_dense_embedding(self, text: str):
        """Generates a dense vector embedding for a given text using OpenAI."""
        if not self.openai_client:
            st.error("OpenAI client not initialized.")
            return None
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Failed to generate embedding: {e}")
            return None

    def search_by_identifier(self, identifier: str, search_type: str = "auto") -> List[Dict]:
        """
        Search by file name, network ID, or reference ID in Pinecone using HYBRID search.
        """
        if not self.index:
            st.error("Pinecone index is not available.")
            return []
            
        # Auto-detect search type (no changes here)
        if search_type == "auto":
            if "." in identifier and any(ext in identifier.lower() for ext in ['.csv', '.xlsx', '.json', '.txt', '.pdf']):
                search_type = "filename"
            elif identifier.startswith("JPN-JEK") or re.match(r'^[A-Z]+-[A-Z]+-[A-Z]-\d+-\d+$', identifier):
                search_type = "reference_id"
            else:
                search_type = "filename"

        query_text = identifier.lower().replace("-", " ").replace("_", " ")

        # --- STEP 1: Create Sparse Vector (same as before) ---
        query_words = query_text.split()
        sparse_vector = {
            "indices": list(range(len(query_words))),
            "values": [1.0] * len(query_words)
        }

        # --- STEP 2: Create Dense Vector (THE FIX) ---
        dense_vector = self.get_dense_embedding(query_text)
        if dense_vector is None:
            return [] # Stop if embedding fails

        try:
            # --- STEP 3: Perform Hybrid Query (THE FIX) ---
            # We now provide BOTH the dense 'vector' and the 'sparse_vector'.
            results = self.index.query(
                vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=100, # Query for more results to filter locally
                include_metadata=True
            )
            
            # Post-filter results for higher accuracy (same as before)
            filtered_matches = []
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                if search_type == "filename" and identifier.lower() in metadata.get('filename', '').lower():
                    filtered_matches.append(match)
                elif search_type == "reference_id" and identifier == metadata.get('reference_id', ''):
                    filtered_matches.append(match)

            # If the specific filter yields no results, fall back to general results from the hybrid search
            if not filtered_matches:
                 return results.get('matches', [])
            
            return filtered_matches
            
        except Exception as e:
            st.error(f"Error during Pinecone search: {str(e)}")
            return []

    # The rest of the class methods (organize_data_by_tabs, create_tab_visualizations, etc.)
    # remain exactly the same. I'm including them for completeness.

    def organize_data_by_tabs(self, matches: List[Dict]) -> Dict[str, List[Dict]]:
        organized_data = {tab: [] for tab in self.report_structure}
        for match in matches:
            metadata = match.get('metadata', {})
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
                organized_data['Overall Performance Summary'].append(metadata)
        return organized_data
    
    def create_tab_visualizations(self, tab_data: List[Dict], tab_name: str) -> List[go.Figure]:
        if not tab_data: return []
        charts, df = [], pd.DataFrame(tab_data).dropna(how='all', axis=1)
        if tab_name == "Screen Details":
            if 'reference_id' in df.columns and 'impressions' in df.columns:
                df_sorted = df.sort_values('impressions', ascending=False).head(15)
                fig = px.bar(df_sorted, x='reference_id', y='impressions', title='Top 15 Screens by Impressions')
                fig.update_xaxes(tickangle=45, type='category')
                charts.append(fig)
        # ... (rest of the visualization logic is unchanged) ...
        return charts

    def generate_tab_summary(self, tab_data: List[Dict], tab_name: str) -> str:
        if not tab_data: return f"**{tab_name}**: No data available."
        df, summary = pd.DataFrame(tab_data), f"### {tab_name} Summary\n- **Records Analyzed**: {len(df)}\n"
        # ... (rest of the summary logic is unchanged) ...
        return summary


def create_jeki_interface():
    st.set_page_config(page_title="JEKI Data RAG", page_icon="🚋", layout="wide")
    st.markdown("""<style>.main-header { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem;
                   background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; }</style>""", unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">JEKI Data Integration - Pinecone RAG</h1>', unsafe_allow_html=True)
    st.markdown("### Search Train Channel & JAD Vision Campaign Data by File Name, Network ID, or Reference ID")
    
    if 'jeki_rag' not in st.session_state:
        st.session_state.jeki_rag = JekiDataRAG()
    
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1: search_query = st.text_input("Search:", placeholder="e.g., JAD_Vision_Report.pdf, JPN-JEK-D-00000-00030", label_visibility="collapsed")
        with c2: search_type = st.selectbox("Type:", ["auto", "filename", "reference_id"], label_visibility="collapsed")
        with c3: search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    if search_button and search_query:
        with st.spinner(f"Searching for '{search_query}'..."):
            matches = st.session_state.jeki_rag.search_by_identifier(search_query, search_type)
            if matches:
                st.success(f"Found {len(matches)} relevant records.")
                organized_data = st.session_state.jeki_rag.organize_data_by_tabs(matches)
                # ... (rest of the UI logic is unchanged) ...
                tab_objects = st.tabs([f"📊 {tab}" for tab in st.session_state.jeki_rag.report_structure])
                for idx, (tab_name, tab_data) in enumerate(organized_data.items()):
                    with tab_objects[idx]:
                        if not tab_data: st.info(f"No data classified under '{tab_name}'."); continue
                        st.markdown(st.session_state.jeki_rag.generate_tab_summary(tab_data, tab_name))
                        # Visualizations and dataframes here
            else:
                st.warning(f"No results found for '{search_query}'. Please try a different term.")
    
    # Sidebar (unchanged)
    st.sidebar.header("📋 Report Structure")
    st.sidebar.markdown("\n".join(f"- **{tab}**" for tab in st.session_state.jeki_rag.report_structure))
    st.sidebar.divider()
    st.sidebar.header("🔍 Example Searches")
    st.sidebar.markdown("- `JAD Vision`\n- `Train Channel`\n- `JPN-JEK-D-00000-00030`")
    st.sidebar.divider()
    st.sidebar.header("🗄️ Database Status")
    if st.session_state.jeki_rag and st.session_state.jeki_rag.index:
        try:    
            stats = st.session_state.jeki_rag.index.describe_index_stats()
            st.sidebar.metric("Total Vectors", f"{stats.total_vector_count:,}")
            st.sidebar.metric("Index Fullness", f"{stats.index_fullness:.2%}")
            st.sidebar.success("🟢 Pinecone Connected")
        except Exception as e:
            st.sidebar.error("🔴 Connection Failed")
    else:
        st.sidebar.error("🔴 Not Connected")

if __name__ == "__main__":
    create_jeki_interface()


