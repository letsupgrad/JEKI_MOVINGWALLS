import streamlit as st
import pinecone
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    st.warning("⚠️ Install sentence-transformers for semantic search: pip install sentence-transformers")

# Page configuration
st.set_page_config(
    page_title="J-AD Vision Analytics Query",
    page_icon="📊",
    layout="wide"
)

# Title and header
st.title("📊 J-AD Vision Campaign Analytics")
st.markdown("Query your Dell campaign data from Pinecone vector database")

# Sidebar for configuration
st.sidebar.header("🔧 Configuration")

# Pinecone configuration
with st.sidebar.expander("Pinecone Settings", expanded=True):
    st.markdown("**Using your Pinecone setup**")
    api_key = st.text_input(
        "Pinecone API Key", 
        value="pcsk_3wbxiS_JFsW8uFyumkQ2oMD5FkfjKJPV5kYkiDwX1T15tg2HtFSn4ioZEeVpsSV6V1DK7s",
        type="password", 
        help="Your API key is pre-filled"
    )
    index_name = st.text_input("Index Name", value="campaign", help="Your index name")
    environment = st.selectbox("Environment", ["us-east-1-aws", "us-west-1-aws", "eu-west-1-aws", "asia-southeast-1-aws"], index=0)
    
    # Add model info
    st.info("🤖 Using multilingual-e5-large embeddings")
    
    if api_key:
        st.success("🔑 API key configured")

# Initialize Pinecone connection
@st.cache_resource
def init_pinecone(api_key, environment):
    try:
        pc = Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {str(e)}")
        return None

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    if EMBEDDINGS_AVAILABLE:
        try:
            model = SentenceTransformer("intfloat/multilingual-e5-large")
            return model
        except Exception as e:
            st.error(f"Failed to load embedding model: {str(e)}")
            return None
    return None

# Query function
def query_pinecone(pc, index_name, query_text, top_k=10):
    try:
        index = pc.Index(index_name)
        
        st.info(f"🔍 Searching for: '{query_text}' in index '{index_name}'")
        
        # Load embedding model for semantic search
        model = load_embedding_model()
        
        if model:
            # Generate embedding for the query
            query_embedding = model.encode(query_text).tolist()
            
            # Perform semantic search
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            st.success(f"✅ Using semantic search with multilingual-e5-large model")
            
        else:
            # Fallback to basic search
            st.warning("🔄 Using fallback search method")
            results = index.query(
                vector=[0.1] * 1024,  # multilingual-e5-large dimension
                top_k=top_k,
                include_metadata=True
            )
        
        return results
        
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        st.error("💡 Check your connection and index configuration")
        return None

# Main interface
if api_key and index_name:
    pc = init_pinecone(api_key, environment)
    
    if pc:
        st.success("✅ Connected to Pinecone successfully!")
        
        # Query section
        st.header("🔍 Query Your Data")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query_input = st.text_input(
                "Enter your query:", 
                value="_JAD Vision 1st April -7th April Dell",
                help="Enter keywords to search your campaign data"
            )
        
        with col2:
            top_k = st.number_input("Results limit", min_value=1, max_value=50, value=10)
        
        if st.button("🔍 Search", type="primary"):
            with st.spinner("Searching your data..."):
                results = query_pinecone(pc, index_name, query_input, top_k)
                
                if results and results.matches:
                    st.success(f"Found {len(results.matches)} results!")
                    
                    # Display results
                    st.header("📋 Search Results")
                    
                    for i, match in enumerate(results.matches):
                        with st.expander(f"Result {i+1} (Score: {match.score:.3f})", expanded=i<3):
                            if match.metadata:
                                # Display metadata in a structured way
                                st.json(match.metadata)
                            else:
                                st.write("No metadata available")
                    
                    # Create visualizations if data is available
                    st.header("📈 Data Visualization")
                    
                    # Sample visualization based on common J-AD Vision metrics
                    # You'll need to adapt this based on your actual data structure
                    try:
                        # Extract data for visualization
                        station_data = []
                        performance_data = []
                        
                        for match in results.matches:
                            if match.metadata:
                                metadata = match.metadata
                                # Adapt these fields based on your actual data structure
                                if 'station_name' in metadata:
                                    station_data.append({
                                        'Station': metadata.get('station_name', 'Unknown'),
                                        'Impressions': metadata.get('impressions', 0),
                                        'Reach': metadata.get('reach', 0)
                                    })
                        
                        if station_data:
                            df_stations = pd.DataFrame(station_data)
                            
                            # Bar chart for station performance
                            fig_bar = px.bar(
                                df_stations, 
                                x='Station', 
                                y='Impressions',
                                title='Impressions by Station',
                                color='Reach'
                            )
                            fig_bar.update_xaxis(tickangle=45)
                            st.plotly_chart(fig_bar, use_container_width=True)
                            
                            # Pie chart for reach distribution
                            if len(df_stations) > 1:
                                fig_pie = px.pie(
                                    df_stations, 
                                    values='Reach', 
                                    names='Station',
                                    title='Reach Distribution by Station'
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                    
                    except Exception as viz_error:
                        st.warning("Could not create visualizations. Raw data displayed above.")
                        st.error(f"Visualization error: {str(viz_error)}")
                
                else:
                    st.warning("No results found for your query. Try different keywords or check your index.")
        
        # Data upload section
        st.header("📤 Upload New Data")
        
        uploaded_file = st.file_uploader("Upload JSON file", type=['json'])
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                st.success("✅ File loaded successfully!")
                
                with st.expander("Preview uploaded data"):
                    st.json(data)
                
                if st.button("💾 Upload to Pinecone"):
                    with st.spinner("Uploading data to Pinecone..."):
                        # Here you would implement the upload logic
                        # This is a placeholder - adapt based on your data structure
                        st.info("Upload functionality would be implemented here based on your specific data structure and embedding strategy.")
                        
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

else:
    st.warning("⚠️ Please configure your Pinecone settings in the sidebar to get started.")

# Footer with information
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Quick Setup")
st.sidebar.markdown("""
**Step 1:** Get your API key from [Pinecone Dashboard](https://app.pinecone.io/)
**Step 2:** Enter "campaign" as index name  
**Step 3:** Search for "_JAD Vision 1st April -7th April Dell"
""")

st.sidebar.markdown("### 📝 About")
st.sidebar.markdown("""
This app allows you to:
- Query J-AD Vision campaign data stored in Pinecone
- Visualize performance metrics
- Upload new campaign data
- Analyze station-wise performance
""")

st.sidebar.markdown("### 🏢 Sample Stations")
st.sidebar.markdown("""
- 新宿駅東口 (Shinjuku East)
- 品川駅中央改札内 (Shinagawa Central)
- 東京駅丸の内地下連絡通路 (Tokyo Marunouchi)
- 横浜駅中央通路 (Yokohama Central)
- 池袋中央改札内 (Ikebukuro Central)
""")

# Add custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)
