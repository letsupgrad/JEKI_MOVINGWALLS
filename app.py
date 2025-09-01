import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from pinecone import Pinecone

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="J-AD Vision Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background: #f0f8f0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .chat-message {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown("""
<div class="main-header">
    <h1>📊 J-AD Vision Campaign Analytics Dashboard</h1>
    <p>Your Dell campaign data is ready! Ask any questions to get insights with tables, charts, and detailed analysis.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar configuration
st.sidebar.header("🔧 System Status")

# Connection status
with st.sidebar.container():
    st.markdown("""
    <div class="status-card">
        <h4>✅ Data Status</h4>
        <p>📅 <strong>Campaign:</strong> Dell J-AD Vision<br>
        📆 <strong>Period:</strong> April 1-7, 2024<br>
        🏢 <strong>Stations:</strong> 25+ locations<br>
        🔗 <strong>Database:</strong> Connected to Pinecone</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize Pinecone connection
@st.cache_resource
def init_pinecone():
    try:
        pc = Pinecone(api_key="pcsk_3wbxiS_JFsW8uFyumkQ2oMD5FkfjKJPV5kYkiDwX1T15tg2HtFSn4ioZEeVpsSV6V1DK7s")
        return pc
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {str(e)}")
        return None

# Load embedding model
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

# Query function with natural language processing
def query_campaign_data(question, top_k=5):
    pc = init_pinecone()
    if not pc:
        return None
    
    model = load_embedding_model()
    if not model:
        st.warning("Semantic search not available. Install sentence-transformers.")
        return None
    
    try:
        index = pc.Index("campaign")
        query_embedding = model.encode(question).tolist()
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return None

# Function to process and visualize data
def create_visualizations(results, question):
    if not results or not results.matches:
        return None
    
    # Extract data from results
    all_data = []
    station_data = []
    daily_data = []
    hourly_data = []
    demographic_data = []
    
    for match in results.matches:
        if match.metadata and 'text' in match.metadata:
            try:
                data = json.loads(match.metadata['text'])
                section = match.metadata.get('section', '')
                
                # Categorize data based on section
                if '新宿' in section or '品川' in section or '東京' in section or '駅' in section:
                    # Station data
                    if isinstance(data, dict):
                        station_info = {
                            'station': section.replace('J・ADビジョン　', ''),
                            'section': section,
                            'data': data,
                            'score': match.score
                        }
                        station_data.append(station_info)
                
                elif 'Daily' in section or '日別' in section:
                    daily_data.append({'section': section, 'data': data, 'score': match.score})
                
                elif 'Hourly' in section or '時間' in section:
                    hourly_data.append({'section': section, 'data': data, 'score': match.score})
                
                elif 'Age' in section or 'Gender' in section or '年齢' in section or '性別' in section:
                    demographic_data.append({'section': section, 'data': data, 'score': match.score})
                
                all_data.append({'section': section, 'data': data, 'score': match.score})
                
            except json.JSONDecodeError:
                continue
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "🏢 Stations", "📅 Daily", "🕐 Hourly", "👥 Demographics"])
    
    with tab1:
        st.subheader("📋 Query Results Summary")
        
        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", len(results.matches))
        with col2:
            avg_score = sum(m.score for m in results.matches) / len(results.matches)
            st.metric("Avg Relevance", f"{avg_score:.2f}")
        with col3:
            st.metric("Station Data", len(station_data))
        with col4:
            high_relevance = sum(1 for m in results.matches if m.score > 0.7)
            st.metric("High Relevance", high_relevance)
        
        # Results table
        st.subheader("🔍 Detailed Results")
        results_df = []
        for i, match in enumerate(results.matches):
            results_df.append({
                'Rank': i + 1,
                'Section': match.metadata.get('section', 'Unknown'),
                'Relevance Score': f"{match.score:.3f}",
                'Content Preview': match.metadata.get('text', '')[:100] + "..." if match.metadata.get('text', '') else "No content"
            })
        
        df = pd.DataFrame(results_df)
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.subheader("🏢 Station Performance")
        
        if station_data:
            # Create station performance chart
            station_names = []
            performance_metrics = []
            
            for station in station_data[:10]:  # Top 10 stations
                station_names.append(station['station'])
                # Extract performance metrics (adapt based on your data structure)
                if isinstance(station['data'], dict):
                    # Look for common metrics
                    impressions = 0
                    reach = 0
                    
                    # Try to extract numerical data
                    for key, value in station['data'].items():
                        if isinstance(value, (int, float)):
                            if 'impression' in key.lower():
                                impressions = value
                            elif 'reach' in key.lower():
                                reach = value
                    
                    performance_metrics.append({'Station': station['station'], 'Impressions': impressions, 'Reach': reach, 'Relevance': station['score']})
            
            if performance_metrics:
                perf_df = pd.DataFrame(performance_metrics)
                
                # Bar chart
                fig_bar = px.bar(perf_df, x='Station', y='Impressions', 
                               title='Station Performance - Impressions',
                               color='Relevance', color_continuous_scale='Viridis')
                fig_bar.update_xaxis(tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Data table
                st.subheader("📊 Station Data Table")
                st.dataframe(perf_df, use_container_width=True)
            
            # Raw station data
            st.subheader("🏢 Station Details")
            for station in station_data[:5]:
                with st.expander(f"📍 {station['station']} (Score: {station['score']:.3f})"):
                    if isinstance(station['data'], dict):
                        st.json(station['data'])
                    else:
                        st.write(station['data'])
        else:
            st.info("No station-specific data found for this query. Try asking about specific stations like '新宿駅' or '品川駅'.")
    
    with tab3:
        st.subheader("📅 Daily Performance")
        
        if daily_data:
            for daily in daily_data:
                st.subheader(f"📊 {daily['section']}")
                if isinstance(daily['data'], dict):
                    # Try to create daily chart
                    if any('day' in str(k).lower() or '日' in str(k) for k in daily['data'].keys()):
                        # Create daily performance visualization
                        daily_df = pd.DataFrame(list(daily['data'].items()), columns=['Date', 'Value'])
                        fig_line = px.line(daily_df, x='Date', y='Value', title='Daily Performance Trend')
                        st.plotly_chart(fig_line, use_container_width=True)
                    
                    st.json(daily['data'])
                else:
                    st.write(daily['data'])
        else:
            st.info("No daily performance data found. Try asking about 'daily performance' or 'day-by-day results'.")
    
    with tab4:
        st.subheader("🕐 Hourly Analysis")
        
        if hourly_data:
            for hourly in hourly_data:
                st.subheader(f"⏰ {hourly['section']}")
                if isinstance(hourly['data'], dict):
                    # Create hourly heatmap or line chart
                    hourly_df = pd.DataFrame(list(hourly['data'].items()), columns=['Hour', 'Value'])
                    fig_hourly = px.bar(hourly_df, x='Hour', y='Value', title='Hourly Performance Distribution')
                    st.plotly_chart(fig_hourly, use_container_width=True)
                    
                    st.json(hourly['data'])
                else:
                    st.write(hourly['data'])
        else:
            st.info("No hourly data found. Try asking about 'hourly performance' or 'peak hours'.")
    
    with tab5:
        st.subheader("👥 Demographics & Audience")
        
        if demographic_data:
            for demo in demographic_data:
                st.subheader(f"👤 {demo['section']}")
                if isinstance(demo['data'], dict):
                    # Create demographic charts
                    demo_df = pd.DataFrame(list(demo['data'].items()), columns=['Category', 'Count'])
                    
                    # Pie chart for demographics
                    fig_pie = px.pie(demo_df, values='Count', names='Category', 
                                   title='Demographic Distribution')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    st.json(demo['data'])
                else:
                    st.write(demo['data'])
        else:
            st.info("No demographic data found. Try asking about 'age groups', 'gender distribution', or 'audience demographics'.")

# Main chat interface
st.header("💬 Ask Questions About Your Campaign")

# Sample questions
st.markdown("### 🎯 Try asking:")
sample_questions = [
    "Show me overall performance summary",
    "Which stations performed best?", 
    "What are the daily trends?",
    "Show demographic breakdown",
    "Compare 新宿駅 vs 品川駅 performance",
    "What were peak hours?",
    "Show me impression data by station"
]

cols = st.columns(3)
for i, question in enumerate(sample_questions):
    col_idx = i % 3
    with cols[col_idx]:
        if st.button(question, key=f"sample_{i}"):
            st.session_state.current_question = question

# Main query input
user_question = st.text_input(
    "🤔 Ask anything about your Dell J-AD Vision campaign:",
    placeholder="e.g., Show me station performance with charts and tables",
    key="main_query"
)

if st.button("🔍 Get Analysis", type="primary") or (hasattr(st.session_state, 'current_question')):
    question = getattr(st.session_state, 'current_question', user_question)
    if hasattr(st.session_state, 'current_question'):
        delattr(st.session_state, 'current_question')
    
    if question:
        with st.spinner(f"🔍 Analyzing: '{question}'..."):
            results = query_campaign_data(question, top_k=10)
            
            if results:
                st.success(f"✅ Found {len(results.matches)} relevant data points!")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'results_count': len(results.matches)
                })
                
                # Create comprehensive analysis
                create_visualizations(results, question)
                
                # Show raw insights
                with st.expander("🔍 Raw Data Insights", expanded=False):
                    for i, match in enumerate(results.matches[:3]):
                        st.subheader(f"Insight {i+1}: {match.metadata.get('section', 'Unknown')}")
                        st.write(f"**Relevance:** {match.score:.3f}")
                        if match.metadata and 'text' in match.metadata:
                            content = match.metadata['text'][:500]
                            st.text_area(f"Content Preview {i+1}", content, height=100)
            else:
                st.error("❌ Unable to retrieve data. Please check your connection.")

# Chat history in sidebar
if st.session_state.chat_history:
    st.sidebar.subheader("💬 Recent Questions")
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
        with st.sidebar.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:30]}..."):
            st.write(f"**Time:** {chat['timestamp']}")
            st.write(f"**Results:** {chat['results_count']} data points")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Overview")
st.sidebar.markdown("""
**Available Data:**
- 📈 Overall performance metrics
- 🏢 25+ station locations  
- 📅 Daily performance (April 1-7)
- 🕐 Hourly breakdowns
- 👥 Age & gender demographics
- 📱 Network summaries
""")

st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Ask specific questions for better results
- Use station names in Japanese (新宿駅)
- Request charts, tables, or comparisons
- Combine multiple metrics in one query
""")
