import streamlit as st
import pinecone
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

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
    api_key = st.text_input("Pinecone API Key", type="password", help="Enter your Pinecone API key")
    index_name = st.text_input("Index Name", value="jad-vision-analytics", help="Name of your Pinecone index")
    environment = st.selectbox("Environment", ["us-east-1-aws", "us-west-1-aws", "eu-west-1-aws", "asia-southeast-1-aws"], index=0)

# Initialize Pinecone connection
@st.cache_resource
def init_pinecone(api_key, environment):
    try:
        pc = Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {str(e)}")
        return None

# Query function
def query_pinecone(pc, index_name, query_text, top_k=10):
    try:
        index = pc.Index(index_name)
        # For text-based querying, you might need to embed the query text
        # This is a simplified version - you may need to adjust based on how you stored the data
        results = index.query(
            vector=[0.1] * 1536,  # Placeholder vector - replace with actual embedding
            top_k=top_k,
            include_metadata=True
        )
        return results
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
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
