# app.py

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime
from pinecone import Pinecone

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# ==============================================================================
# FUNCTION DEFINITIONS
# ==============================================================================

@st.cache_resource
def init_pinecone():
    """Initializes and returns the Pinecone client."""
    try:
        # --- MODIFICATION: PASTE YOUR API KEY HERE ---
        # WARNING: Do not share this code publicly with your key included.
        PINECONE_API_KEY = "YOUR_PINECONE_API_KEY_HERE" 
        
        if PINECONE_API_KEY == "YOUR_PINECONE_API_KEY_HERE":
            st.error("Please replace 'YOUR_PINECONE_API_KEY_HERE' with your actual Pinecone API key.")
            return None
            
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {str(e)}")
        return None

@st.cache_resource
def load_embedding_model():
    """Loads and caches the sentence transformer model."""
    if EMBEDDINGS_AVAILABLE:
        try:
            # Using a powerful multilingual model suitable for Japanese and English
            model = SentenceTransformer("intfloat/multilingual-e5-large")
            return model
        except Exception as e:
            st.error(f"Failed to load embedding model: {str(e)}")
            return None
    st.warning("Embeddings are not available. Please install `sentence-transformers`.")
    return None

def get_section_specific_query(pc, index_name, section_name):
    """Queries for a specific section by name for high-precision retrieval."""
    if not pc or not EMBEDDINGS_AVAILABLE:
        return None
    model = load_embedding_model()
    if not model:
        return None
    
    try:
        index = pc.Index(index_name)
        query_embedding = model.encode(section_name).tolist()
        
        results = index.query(
            vector=query_embedding,
            top_k=5, # Fetch a few related items as well
            include_metadata=True
        )
        
        # Prioritize exact matches in metadata
        filtered_matches = [
            match for match in results.matches
            if match.metadata and 'section' in match.metadata and
               (section_name in match.metadata['section'] or match.metadata['section'] in section_name)
        ]
        
        # If exact matches are found, use them; otherwise, return the semantic results
        if filtered_matches:
            results.matches = sorted(filtered_matches, key=lambda x: x.score, reverse=True)
        return results
                
    except Exception as e:
        st.error(f"Section-specific query failed: {str(e)}")
        return None

def query_campaign_data(question, top_k=10):
    """Performs a semantic search on the campaign data using embeddings."""
    pc = init_pinecone()
    model = load_embedding_model()
    if not pc or not model:
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

def create_visualizations(results, question):
    """Processes retrieved data and creates appropriate visualizations in tabs."""
    if not results or not results.matches:
        st.warning("No data found to create visualizations.")
        return

    # Extract and categorize data from results
    station_data, daily_data, hourly_data, demographic_data = [], [], [], []
    
    for match in results.matches:
        if match.metadata and 'text' in match.metadata:
            try:
                data = json.loads(match.metadata['text'])
                section = match.metadata.get('section', '')
                
                if any(kw in section for kw in ['新宿', '品川', '東京', '駅', '横浜', '池袋', '恵比寿', '上野', '秋葉原']):
                    station_data.append({'station': section.replace('J・ADビジョン\u3000', ''), 'data': data, 'score': match.score})
                elif 'Daily' in section or '日別' in section:
                    daily_data.append({'section': section, 'data': data, 'score': match.score})
                elif 'Hourly' in section or '時間' in section:
                    hourly_data.append({'section': section, 'data': data, 'score': match.score})
                elif 'Age' in section or 'Gender' in section or '年齢' in section or '性別' in section:
                    demographic_data.append({'section': section, 'data': data, 'score': match.score})
            except (json.JSONDecodeError, TypeError):
                continue
    
    st.header(f"🔎 Data Exploration for: '{question}'")
    tabs = st.tabs(["📊 Summary", "🏢 Stations", "📅 Daily", "🕐 Hourly", "👥 Demographics"])
    
    with tabs[0]:
        st.subheader("📋 Retrieved Data Summary")
        results_df_data = [{
            'Rank': i + 1,
            'Section': match.metadata.get('section', f'Item {i+1}').replace('J・ADビジョン\u3000', ''),
            'Relevance Score': f"{match.score:.3f}",
            'Content Preview': (match.metadata.get('text', '')[:200] + '...') if match.metadata.get('text') else "N/A"
        } for i, match in enumerate(results.matches)]
        st.dataframe(pd.DataFrame(results_df_data), use_container_width=True)

    with tabs[1]:
        st.subheader("🏢 Station Performance")
        if station_data:
            performance_metrics = []
            for station in station_data:
                impressions = next((v for k, v in station.get('data', {}).items() if 'impression' in str(k).lower() and isinstance(v, (int, float))), 0)
                performance_metrics.append({'Station': station['station'], 'Impressions': impressions, 'Relevance': station['score']})
            
            perf_df = pd.DataFrame(performance_metrics).sort_values('Impressions', ascending=False).drop_duplicates(subset=['Station'])
            if not perf_df.empty:
                fig_bar = px.bar(perf_df, x='Station', y='Impressions', title='Station Performance - Impressions', color='Relevance', color_continuous_scale='Viridis')
                st.plotly_chart(fig_bar, use_container_width=True)
                st.dataframe(perf_df, use_container_width=True)
        else:
            st.info("No station-specific data was found for this query.")

    with tabs[2]:
        st.subheader("📅 Daily Performance")
        if daily_data:
            daily_item = daily_data[0]
            if isinstance(daily_item['data'], dict):
                daily_df = pd.DataFrame(list(daily_item['data'].items()), columns=['Date', 'Value'])
                fig_line = px.line(daily_df, x='Date', y='Value', title=f'Daily Trend: {daily_item["section"]}', markers=True)
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No daily performance data was found for this query.")

    with tabs[3]:
        st.subheader("🕐 Hourly Analysis")
        if hourly_data:
            hourly_item = hourly_data[0]
            if isinstance(hourly_item['data'], dict):
                hourly_df = pd.DataFrame(list(hourly_item['data'].items()), columns=['Hour', 'Value'])
                fig_hourly = px.bar(hourly_df, x='Hour', y='Value', title=f'Hourly Distribution: {hourly_item["section"]}')
                st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.info("No hourly data was found for this query.")

    with tabs[4]:
        st.subheader("👥 Demographics & Audience")
        if demographic_data:
            demo_item = demographic_data[0]
            if isinstance(demo_item['data'], dict):
                demo_df = pd.DataFrame(list(demo_item['data'].items()), columns=['Category', 'Count'])
                fig_pie = px.pie(demo_df, values='Count', names='Category', title=f'Demographic Distribution: {demo_item["section"]}')
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No demographic data was found for this query.")

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="J-AD Vision Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        color: white; padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;
    }
    .status-card {
        background: #f0f8f0; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MAIN APP LAYOUT
# ==============================================================================

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title and header
st.markdown("""
<div class="main-header">
    <h1>📊 J-AD Vision Campaign Analytics Dashboard</h1>
    <p>Your Dell campaign data is ready! Browse all available sections or ask questions to get insights with tables, charts, and detailed analysis.</p>
</div>
""", unsafe_allow_html=True)

# Data catalog section
st.header("📁 Available Data Sections")
available_sections = {
    "📋 Report Information": ["Report Info", "Report Info - JPN"],
    "📖 Glossary & Notes": ["Glossary & Note", "Glossary & Note - JPN"],
    "🖥️ Screen Details": ["Screen Details"],
    "📊 Performance Analytics": ["1. Overall performance summary", "2. Daily", "3. Overall Age and Gender", "4. Overall Hourly", "5. Network Summary"],
    "🏢 Station Locations - Tokyo Central": ["J・ADビジョン\u3000東京駅丸の内地下連絡通路", "J・ADビジョン\u3000東京駅京葉通路", "J・ADビジョン\u3000東京駅新幹線北乗換口", "J・ADビジョン\u3000東京駅新幹線南乗換口", "J・ADビジョン\u3000有楽町駅中央改札口"],
    "🏢 Station Locations - Shinjuku Area": ["J・ADビジョン\u3000新宿駅東口", "J・ADビジョン\u3000新宿駅南口", "J・ADビジョン\u3000新宿駅甲州街道改札"],
    "🏢 Station Locations - Major Hubs": ["J・ADビジョン\u3000品川駅中央改札内", "J・ADビジョン\u3000横浜駅中央通路", "J・ADビジョン\u3000横浜駅南改札内", "J・ADビジョン\u3000JR横浜タワーアトリウム", "J・ADビジョン\u3000池袋中央改札内"],
    "🏢 Station Locations - Other Lines": ["J・ADビジョン\u3000巣鴨駅改札外", "J・ADビジョン\u3000五反田駅", "J・ADビジョン\u3000高輪ゲートウェイ駅", "J・ADビジョン\u3000秋葉原駅新電気街口", "J・ADビジョン\u3000吉祥寺駅南北自由通路", "J・ADビジョン\u3000浦和駅改札口", "J・ADビジョン\u3000大宮駅中央改札", "J・ADビジョン\u3000高田馬場駅スマイル・ステーションビジョン", "J・ADビジョン\u3000桜木町駅", "J・ADビジョン\u3000恵比寿駅西口", "J・ADビジョン\u3000赤羽駅北改札", "J・ADビジョン\u3000八王子駅自由通路南", "J・ADビジョン\u3000上野駅公園改札内"]
}

col1, col2 = st.columns([2, 1])
with col1:
    for category, sections in available_sections.items():
        with st.expander(f"{category} ({len(sections)} items)"):
            for section in sections:
                if st.button(f"➡️ View {section.replace('J・ADビジョン\u3000', '')}", key=f"view_{section}"):
                    st.session_state.selected_section = section
                    st.rerun()
with col2:
    st.subheader("📊 Quick Stats")
    st.metric("📁 Total Sections", sum(len(s) for s in available_sections.values()))
    st.metric("🏢 Station Locations", sum(len(s) for c, s in available_sections.items() if "Station" in c))
    st.metric("📈 Analytics Reports", 5)

# Handle section selection from data catalog
if 'selected_section' in st.session_state and st.session_state.selected_section:
    selected_query = st.session_state.selected_section
    st.info(f"💡 Browsing section: '{selected_query}'")
    
    with st.spinner(f"🔍 Loading data for: {selected_query}..."):
        pinecone_client = init_pinecone()
        if pinecone_client:
            results = get_section_specific_query(pinecone_client, "campaign", selected_query)
            if results and results.matches:
                st.success(f"✅ Found detailed data for {st.session_state.selected_section}")
                create_visualizations(results, selected_query)
            else:
                st.warning(f"❌ No specific data found for {st.session_state.selected_section}")
        
    if st.button("🔄 Clear Selection"):
        st.session_state.selected_section = None
        st.rerun()

# Main chat interface
st.header("💬 Ask Questions About Your Campaign")
st.markdown("##### 🎯 Try asking:")
sample_questions = ["Show me overall performance summary", "Which stations performed best?", "What are the daily trends?", "Show demographic breakdown", "Compare 新宿駅 vs 品川駅 performance", "What were peak hours?"]
cols = st.columns(3)
for i, question in enumerate(sample_questions):
    if cols[i % 3].button(question, key=f"sample_{i}"):
        st.session_state.current_question = question
        st.rerun()

user_question = st.text_input(
    "🤔 Ask anything about your Dell J-AD Vision campaign:",
    placeholder="e.g., Show me station performance with charts and tables",
    key="main_query"
)

# Process query from either button click or text input
query_to_process = st.session_state.pop('current_question', None) or (user_question if st.button("🔍 Get Analysis", type="primary") else None)

if query_to_process:
    with st.spinner(f"🔍 Searching for data related to: '{query_to_process}'..."):
        results = query_campaign_data(query_to_process)
        
        if results and results.matches:
            st.success(f"✅ Found {len(results.matches)} relevant data points!")
            
            # Add to chat history
            st.session_state.chat_history.append({'question': query_to_process, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'results_count': len(results.matches)})
            
            # Directly create visualizations
            create_visualizations(results, query_to_process)
        else:
            st.error("❌ Unable to retrieve data for your question. Please try rephrasing it.")

# Sidebar & Footer
st.sidebar.header("🔧 System Status")
st.sidebar.markdown(f"""
<div class="status-card">
    <h4>✅ System Status</h4>
    <p><strong>Pinecone DB:</strong> {'Connected' if init_pinecone() else 'Error'}<br>
    <strong>Embedding Model:</strong> {'Loaded' if EMBEDDINGS_AVAILABLE else 'Not Available'}</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.chat_history:
    st.sidebar.subheader("💬 Recent Questions")
    for chat in reversed(st.session_state.chat_history[-5:]):
        st.sidebar.info(f"**Q:** {chat['question'][:40]}...\n**A:** Found {chat['results_count']} results")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Ask specific questions for better results.
- Use station names in Japanese (e.g., 新宿駅).
- Request charts, tables, or comparisons.
""")
