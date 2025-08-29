import streamlit as st
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List, Dict
import json
import time

# Initialize session state
if 'pinecone_client' not in st.session_state:
    st.session_state.pinecone_client = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None

# Campaign data from the documents
CAMPAIGN_DATA = {
    "jad_vision_campaign": {
        "duration": "March 4, 2024 to March 10, 2024",
        "spot_duration": "15-second ad spots in a single loop",
        "total_reference_ids": 25,
        "overall_performance": {
            "total_impressions": 1021333,  # Sum of all individual location impressions
            "total_reach": 803313,
            "average_frequency": 1.27
        },
        "locations": [
            {"id": "JPN-JEK-D-00000-00029", "name": "Sugamo Station Exit", "impressions": 49682, "frequency": 1.66, "reach": 29929},
            {"id": "JPN-JEK-D-00000-00030", "name": "Shinjuku Station East Exit", "impressions": 61259, "frequency": 1.11, "reach": 55188},
            {"id": "JPN-JEK-D-00000-00031", "name": "Shinjuku Station South Exit", "impressions": 63881, "frequency": 1.00, "reach": 63881},
            {"id": "JPN-JEK-D-00000-00032", "name": "Shinjuku Station Koshu-Kaido Exit", "impressions": 50093, "frequency": 1.17, "reach": 42815},
            {"id": "JPN-JEK-D-00000-00033", "name": "Shibuya Station Hachiko Exit", "impressions": 53843, "frequency": 1.22, "reach": 44134},
            {"id": "JPN-JEK-D-00000-00034", "name": "Gotanda Station", "impressions": 21828, "frequency": 1.46, "reach": 14951},
            {"id": "JPN-JEK-D-00000-00035", "name": "Shinagawa Station Central Gate", "impressions": 9342, "frequency": 1.05, "reach": 8897},
            {"id": "JPN-JEK-D-00000-00036", "name": "Takanawa Gateway Station", "impressions": 3669, "frequency": 1.60, "reach": 2293},
            {"id": "JPN-JEK-D-00000-00039", "name": "Yurakucho Station Central Gate", "impressions": 13464, "frequency": 1.16, "reach": 11607},
            {"id": "JPN-JEK-D-00000-00040", "name": "Tokyo Station Marunouchi Underground Passage", "impressions": 19369, "frequency": 1.23, "reach": 15747},
            {"id": "JPN-JEK-D-00000-00041", "name": "Tokyo Station Keiyo Passage", "impressions": 17793, "frequency": 1.20, "reach": 14828},
            {"id": "JPN-JEK-D-00000-00042", "name": "Akihabara Station New Electric Town Exit", "impressions": 50248, "frequency": 1.13, "reach": 44467},
            {"id": "JPN-JEK-D-00000-00044", "name": "Kichijoji Station North-South Free Passage", "impressions": 14756, "frequency": 1.22, "reach": 12095},
            {"id": "JPN-JEK-D-00000-00045", "name": "Urawa Station Gate", "impressions": 53167, "frequency": 1.35, "reach": 39383},
            {"id": "JPN-JEK-D-00000-00046", "name": "Omiya Station Central Gate", "impressions": 43421, "frequency": 1.42, "reach": 30578},
            {"id": "JPN-JEK-D-00000-00047", "name": "Yokohama Station Central Passage", "impressions": 103351, "frequency": 1.16, "reach": 89096},
            {"id": "JPN-JEK-D-00000-00048", "name": "JR Yokohama Tower Atrium", "impressions": 27946, "frequency": 1.06, "reach": 26364},
            {"id": "JPN-JEK-D-00000-00049", "name": "Takadanobaba Station Smile Vision", "impressions": 51685, "frequency": 1.35, "reach": 38285},
            {"id": "JPN-JEK-D-00000-00050", "name": "Ikebukuro Station Central Gate", "impressions": 107227, "frequency": 1.19, "reach": 90107},
            {"id": "JPN-JEK-D-00000-00051", "name": "Sakuragicho Station", "impressions": 32180, "frequency": 1.38, "reach": 23319},
            {"id": "JPN-JEK-D-00000-00052", "name": "Yokohama Station South Gate", "impressions": 69742, "frequency": 1.43, "reach": 48771},
            {"id": "JPN-JEK-D-00000-00058", "name": "Tokyo Station Shinkansen North Transfer Gate", "impressions": 4725, "frequency": 1.00, "reach": 4725},
            {"id": "JPN-JEK-D-00000-00059", "name": "Tokyo Station Shinkansen South Transfer Gate", "impressions": 4514, "frequency": 1.00, "reach": 4514},
            {"id": "JPN-JEK-D-00000-00060", "name": "Ebisu Station West Exit", "impressions": 62157, "frequency": 1.04, "reach": 59766},
            {"id": "JPN-JEK-D-00000-00061", "name": "Akabane Station North Gate", "impressions": 31991, "frequency": 1.01, "reach": 31674}
        ],
        "demographics": {
            "age_distribution": {
                "10-19": {"percentage": 7.66, "impressions": 78281},
                "20-29": {"percentage": 16.59, "impressions": 169464},
                "30-39": {"percentage": 18.57, "impressions": 189680},
                "40-49": {"percentage": 22.47, "impressions": 229510},
                "50-59": {"percentage": 21.97, "impressions": 224345},
                "60+": {"percentage": 12.73, "impressions": 130053}
            },
            "gender_distribution": {
                "male": {"percentage": 59.56, "impressions": 608321},
                "female": {"percentage": 40.44, "impressions": 413012}
            }
        }
    }
}

def init_pinecone():
    """Initialize Pinecone connection"""
    try:
        api_key = st.session_state.get('pinecone_api_key', '')
        if not api_key:
            st.error("Please enter your Pinecone API key")
            return None
        
        pc = Pinecone(api_key=api_key)
        st.session_state.pinecone_client = pc
        return pc
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

def create_or_get_index(pc, index_name="campaign-data"):
    """Create or get existing Pinecone index"""
    try:
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [idx['name'] for idx in existing_indexes]
        
        if index_name not in index_names:
            # Create new index
            pc.create_index(
                name=index_name,
                dimension=384,  # For sentence-transformers/all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            st.success(f"Created new index: {index_name}")
        
        index = pc.Index(index_name)
        st.session_state.index = index
        return index
    except Exception as e:
        st.error(f"Error with index: {str(e)}")
        return None

def init_embedder():
    """Initialize sentence transformer model"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.session_state.embedder = model
        return model
    except Exception as e:
        st.error(f"Error loading embedder: {str(e)}")
        return None

def create_embeddings_and_upsert(index, embedder):
    """Create embeddings from campaign data and upsert to Pinecone"""
    try:
        vectors = []
        
        # Create embeddings for different aspects of the data
        data_points = [
            {
                "id": "campaign_overview",
                "text": f"JAD Vision campaign duration {CAMPAIGN_DATA['jad_vision_campaign']['duration']} with {CAMPAIGN_DATA['jad_vision_campaign']['spot_duration']} and {CAMPAIGN_DATA['jad_vision_campaign']['total_reference_ids']} reference IDs",
                "metadata": {"type": "overview", "category": "campaign_info"}
            },
            {
                "id": "overall_performance",
                "text": f"Total impressions {CAMPAIGN_DATA['jad_vision_campaign']['overall_performance']['total_impressions']}, total reach {CAMPAIGN_DATA['jad_vision_campaign']['overall_performance']['total_reach']}, average frequency {CAMPAIGN_DATA['jad_vision_campaign']['overall_performance']['average_frequency']}",
                "metadata": {"type": "performance", "category": "overall_metrics"}
            },
            {
                "id": "demographics_age",
                "text": f"Age distribution: 40-49 years highest at 22.47% with 229,510 impressions, followed by 50-59 years at 21.97% with 224,345 impressions, 30-39 years at 18.57% with 189,680 impressions",
                "metadata": {"type": "demographics", "category": "age"}
            },
            {
                "id": "demographics_gender",
                "text": f"Gender distribution: Male 59.56% with 608,321 impressions, Female 40.44% with 413,012 impressions",
                "metadata": {"type": "demographics", "category": "gender"}
            }
        ]
        
        # Add location-specific data
        for location in CAMPAIGN_DATA['jad_vision_campaign']['locations']:
            data_points.append({
                "id": f"location_{location['id']}",
                "text": f"Reference ID {location['id']} {location['name']} has {location['impressions']} impressions, frequency {location['frequency']}, reach {location['reach']}",
                "metadata": {
                    "type": "location", 
                    "category": "station_performance",
                    "reference_id": location['id'],
                    "station_name": location['name'],
                    "impressions": location['impressions']
                }
            })
        
        # Create embeddings
        for dp in data_points:
            embedding = embedder.encode(dp['text']).tolist()
            vectors.append({
                "id": dp['id'],
                "values": embedding,
                "metadata": dp['metadata']
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        st.success(f"Successfully upserted {len(vectors)} vectors to Pinecone")
        return True
        
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return False

def query_rag(question: str, index, embedder, top_k=5):
    """Query the RAG system"""
    try:
        # Create embedding for the question
        question_embedding = embedder.encode(question).tolist()
        
        # Query Pinecone
        results = index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract relevant information
        context = []
        for match in results['matches']:
            score = match['score']
            metadata = match.get('metadata', {})
            
            # Build context based on metadata
            if metadata.get('type') == 'location':
                context.append(f"Location: {metadata.get('station_name', 'Unknown')} (ID: {metadata.get('reference_id', 'Unknown')}) - {metadata.get('impressions', 0)} impressions (Score: {score:.3f})")
            elif metadata.get('type') == 'performance':
                context.append(f"Overall Performance Data (Score: {score:.3f})")
            elif metadata.get('type') == 'demographics':
                context.append(f"Demographics - {metadata.get('category', 'Unknown')} data (Score: {score:.3f})")
            elif metadata.get('type') == 'overview':
                context.append(f"Campaign Overview Information (Score: {score:.3f})")
        
        return results, context
        
    except Exception as e:
        st.error(f"Error querying RAG: {str(e)}")
        return None, []

def generate_answer(question: str, context: List[str], matches):
    """Generate answer based on retrieved context"""
    try:
        # Extract detailed information from matches
        detailed_info = []
        
        for match in matches['matches']:
            metadata = match.get('metadata', {})
            score = match['score']
            
            if score > 0.7:  # High relevance threshold
                if metadata.get('type') == 'location':
                    ref_id = metadata.get('reference_id')
                    # Find detailed info from our data
                    for loc in CAMPAIGN_DATA['jad_vision_campaign']['locations']:
                        if loc['id'] == ref_id:
                            detailed_info.append({
                                "type": "location",
                                "data": loc
                            })
                elif metadata.get('type') == 'performance':
                    detailed_info.append({
                        "type": "performance",
                        "data": CAMPAIGN_DATA['jad_vision_campaign']['overall_performance']
                    })
                elif metadata.get('type') == 'demographics':
                    detailed_info.append({
                        "type": "demographics",
                        "data": CAMPAIGN_DATA['jad_vision_campaign']['demographics']
                    })
                elif metadata.get('type') == 'overview':
                    detailed_info.append({
                        "type": "overview",
                        "data": {
                            "duration": CAMPAIGN_DATA['jad_vision_campaign']['duration'],
                            "spot_duration": CAMPAIGN_DATA['jad_vision_campaign']['spot_duration'],
                            "total_reference_ids": CAMPAIGN_DATA['jad_vision_campaign']['total_reference_ids']
                        }
                    })
        
        return detailed_info
        
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return []

def format_response(question: str, detailed_info: List[Dict]):
    """Format the response based on retrieved information"""
    if not detailed_info:
        return "No relevant information found for your question."
    
    response = f"**Answer to: {question}**\n\n"
    
    for info in detailed_info:
        if info['type'] == 'location':
            loc = info['data']
            response += f"📍 **{loc['name']}** (ID: {loc['id']})\n"
            response += f"   • Impressions: {loc['impressions']:,}\n"
            response += f"   • Reach: {loc['reach']:,}\n"
            response += f"   • Frequency: {loc['frequency']}\n\n"
            
        elif info['type'] == 'performance':
            perf = info['data']
            response += f"📊 **Overall Performance**\n"
            response += f"   • Total Impressions: {perf['total_impressions']:,}\n"
            response += f"   • Total Reach: {perf['total_reach']:,}\n"
            response += f"   • Average Frequency: {perf['average_frequency']}\n\n"
            
        elif info['type'] == 'demographics':
            demo = info['data']
            response += f"👥 **Demographics**\n"
            response += f"   **Gender Distribution:**\n"
            for gender, data in demo['gender_distribution'].items():
                response += f"   • {gender.title()}: {data['percentage']:.2f}% ({data['impressions']:,} impressions)\n"
            response += f"\n   **Age Distribution:**\n"
            for age, data in demo['age_distribution'].items():
                response += f"   • {age} years: {data['percentage']:.2f}% ({data['impressions']:,} impressions)\n"
            response += "\n"
            
        elif info['type'] == 'overview':
            overview = info['data']
            response += f"📅 **Campaign Overview**\n"
            response += f"   • Duration: {overview['duration']}\n"
            response += f"   • Spot Duration: {overview['spot_duration']}\n"
            response += f"   • Total Reference IDs: {overview['total_reference_ids']}\n\n"
    
    return response

def main():
    st.set_page_config(
        page_title="JAD Vision Campaign RAG System",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 JAD Vision Campaign RAG System")
    st.markdown("Query campaign data using Pinecone vector database and RAG")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Pinecone API Key
        pinecone_api_key = st.text_input(
            "Pinecone API Key",
            type="password",
            help="Enter your Pinecone API key"
        )
        
        if pinecone_api_key:
            st.session_state.pinecone_api_key = pinecone_api_key
        
        # Initialize button
        if st.button("🚀 Initialize RAG System"):
            with st.spinner("Initializing..."):
                # Initialize Pinecone
                pc = init_pinecone()
                if pc:
                    # Create/get index
                    index = create_or_get_index(pc)
                    if index:
                        # Initialize embedder
                        embedder = init_embedder()
                        if embedder:
                            # Create embeddings and upsert
                            success = create_embeddings_and_upsert(index, embedder)
                            if success:
                                st.success("✅ RAG system initialized successfully!")
                            else:
                                st.error("❌ Failed to initialize RAG system")
        
        st.markdown("---")
        st.header("📋 Quick Stats")
        st.metric("Total Locations", 25)
        st.metric("Campaign Duration", "7 days")
        st.metric("Total Impressions", "1,021,333")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤖 Ask Questions About Campaign Data")
        
        # Sample questions
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What are the total impressions?",
            "Which location had the highest impressions?",
            "Show me demographics breakdown",
            "What is the performance of Ikebukuro Station?",
            "List all reference IDs",
            "What was the campaign duration?",
            "Show gender distribution",
            "Which age group had the most impressions?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            col = cols[i % 2]
            if col.button(f"❓ {question}", key=f"sample_{i}"):
                st.session_state.current_question = question
        
        # Question input
        question = st.text_input(
            "🔍 Enter your question:",
            value=st.session_state.get('current_question', ''),
            placeholder="e.g., What are the impressions for Shinjuku Station?"
        )
        
        if st.button("🔎 Search", type="primary"):
            if question and st.session_state.index and st.session_state.embedder:
                with st.spinner("Searching..."):
                    matches, context = query_rag(
                        question, 
                        st.session_state.index, 
                        st.session_state.embedder
                    )
                    
                    if matches:
                        detailed_info = generate_answer(question, context, matches)
                        response = format_response(question, detailed_info)
                        
                        st.markdown("### 📋 Results")
                        st.markdown(response)
                        
                        # Show similarity scores
                        with st.expander("🔍 Similarity Scores"):
                            for i, match in enumerate(matches['matches']):
                                st.write(f"**Match {i+1}:** Score: {match['score']:.3f}")
                                st.write(f"Metadata: {match.get('metadata', {})}")
                    else:
                        st.warning("No results found. Please check your Pinecone connection.")
            else:
                if not question:
                    st.warning("Please enter a question")
                else:
                    st.warning("Please initialize the RAG system first")
    
    with col2:
        st.header("📈 Campaign Statistics")
        
        # Top performing locations
        locations_df = pd.DataFrame(CAMPAIGN_DATA['jad_vision_campaign']['locations'])
        top_locations = locations_df.nlargest(5, 'impressions')
        
        st.markdown("### 🏆 Top 5 Locations by Impressions")
        for _, loc in top_locations.iterrows():
            st.metric(
                label=loc['name'][:25] + "..." if len(loc['name']) > 25 else loc['name'],
                value=f"{loc['impressions']:,}",
                delta=f"Reach: {loc['reach']:,}"
            )
        
        # Demographics chart
        st.markdown("### 👥 Age Distribution")
        age_data = CAMPAIGN_DATA['jad_vision_campaign']['demographics']['age_distribution']
        age_df = pd.DataFrame([
            {"Age Group": k, "Impressions": v['impressions'], "Percentage": v['percentage']}
            for k, v in age_data.items()
        ])
        st.bar_chart(age_df.set_index('Age Group')['Impressions'])
        
        # Gender distribution
        st.markdown("### ⚥ Gender Distribution")
        gender_data = CAMPAIGN_DATA['jad_vision_campaign']['demographics']['gender_distribution']
        gender_df = pd.DataFrame([
            {"Gender": k.title(), "Impressions": v['impressions'], "Percentage": v['percentage']}
            for k, v in gender_data.items()
        ])
        st.bar_chart(gender_df.set_index('Gender')['Impressions'])
    
    # System status
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Connected" if st.session_state.pinecone_client else "❌ Not Connected"
        st.metric("Pinecone Status", status)
    
    with col2:
        status = "✅ Ready" if st.session_state.index else "❌ Not Ready"
        st.metric("Index Status", status)
    
    with col3:
        status = "✅ Loaded" if st.session_state.embedder else "❌ Not Loaded"
        st.metric("Embedder Status", status)
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Setup Instructions:**
        1. Enter your Pinecone API key in the sidebar
        2. Click "Initialize RAG System" to set up the vector database
        3. Wait for the system to load campaign data into Pinecone
        
        **Usage:**
        1. Use the sample questions or type your own query
        2. Click "Search" to get AI-powered answers
        3. View detailed results with similarity scores
        
        **Available Data:**
        - Campaign overview and duration
        - Location-specific impressions and performance
        - Demographics (age and gender distribution)
        - Reference ID details
        - Overall performance metrics
        
        **Requirements:**
        ```bash
        pip install streamlit pinecone-client sentence-transformers pandas numpy
        ```
        """)

if __name__ == "__main__":
    main()
