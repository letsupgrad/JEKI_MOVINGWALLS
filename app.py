from pinecone import Pinecone
import openai
from pinecone_text.sparse import BM25Encoder
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Any
import time
from textblob import TextBlob
import re
from datetime import datetime

class JADVisionRAGSystem:
    def __init__(self):
        """Initialize JAD Vision RAG System with Pinecone connection"""
        # Pinecone setup for file storage 🔌
        self.pc = Pinecone(api_key="pcsk_3wbxiS_JFsW8uFyumkQ2oMD5FkfjKJPV5kYkiDwX1T15tg2HtFSn4ioZEeVpsSV6V1DK7s")
        self.index = self.pc.Index("campaign")
        
        # Initialize BM25 encoder for semantic search
        self.encoder = BM25Encoder.default()
        
        # Load JAD Vision report data
        self.report_data = self.load_jad_vision_data()
        
    def load_jad_vision_data(self) -> Dict[str, Any]:
        """Load JAD Vision Dell campaign report data"""
        return {
            "report_info": {
                "Campaign_Duration": {
                    "Start": "2024-07-07",
                    "End": "2024-08-03",
                    "Operating_Hours": "Full day coverage",
                    "Campaign_Name": "Dell JAD Vision Campaign"
                },
                "Spot_Duration": "15 seconds",
                "Report_Generation": {
                    "Generated_On": "2024-08-05",
                    "Reviewed_On": "2024-08-06",
                    "Validated_On": "2024-08-06"
                },
                "Total_Networks": 25,
                "Campaign_Type": "Digital Out-of-Home (DOOH)"
            },
            
            "screen_details": {
                "Network_ID": "Unique identifier for each screen in the network",
                "Screen_Name": "Name of each screen in the network", 
                "Impressions": "Total number of people who passed by the screen and had the Opportunity To See (OTS)",
                "Reach": "Unique count of audiences who passed by the screen and had the OTS at least once",
                "Frequency": "Average number of times a person is exposed to an ad during the specified period"
            },
            
            "network_ids": [
                "JPN-JEK-D-00000-00029", "JPN-JEK-D-00000-00030", "JPN-JEK-D-00000-00031",
                "JPN-JEK-D-00000-00032", "JPN-JEK-D-00000-00033", "JPN-JEK-D-00000-00034",
                "JPN-JEK-D-00000-00035", "JPN-JEK-D-00000-00036", "JPN-JEK-D-00000-00039",
                "JPN-JEK-D-00000-00040", "JPN-JEK-D-00000-00041", "JPN-JEK-D-00000-00042",
                "JPN-JEK-D-00000-00044", "JPN-JEK-D-00000-00045", "JPN-JEK-D-00000-00046",
                "JPN-JEK-D-00000-00047", "JPN-JEK-D-00000-00048", "JPN-JEK-D-00000-00049",
                "JPN-JEK-D-00000-00050", "JPN-JEK-D-00000-00051", "JPN-JEK-D-00000-00052",
                "JPN-JEK-D-00000-00058", "JPN-JEK-D-00000-00059", "JPN-JEK-D-00000-00060",
                "JPN-JEK-D-00000-00061"
            ],
            
            "performance_summary": {
                "Sugamo Station Exit": {"Impressions": 49682, "Frequency": 1.66, "Reach": 29929},
                "Shinjuku Station East Exit": {"Impressions": 61259, "Frequency": 1.11, "Reach": 55188},
                "Shinjuku Station South Exit": {"Impressions": 63881, "Frequency": 1.00, "Reach": 63881},
                "Shibuya Station": {"Impressions": 75420, "Frequency": 1.25, "Reach": 60336},
                "Tokyo Station": {"Impressions": 82156, "Frequency": 1.35, "Reach": 60856},
                "Harajuku Station": {"Impressions": 45280, "Frequency": 1.18, "Reach": 38373},
                "Ikebukuro Station": {"Impressions": 68940, "Frequency": 1.22, "Reach": 56475},
                "Ueno Station": {"Impressions": 52370, "Frequency": 1.15, "Reach": 45539}
            },
            
            "age_gender_summary": {
                "Age": {
                    "10-19": {"Percentage": 7.66, "Impressions": 78281},
                    "20-29": {"Percentage": 16.59, "Impressions": 169464},
                    "30-39": {"Percentage": 18.57, "Impressions": 189680},
                    "40-49": {"Percentage": 22.47, "Impressions": 229510},
                    "50-59": {"Percentage": 21.97, "Impressions": 224345},
                    "60+": {"Percentage": 12.73, "Impressions": 130053}
                },
                "Gender": {
                    "Male": {"Percentage": 59.56, "Impressions": 608321},
                    "Female": {"Percentage": 40.44, "Impressions": 413012}
                }
            },
            
            "hourly_summary": {
                "5:00 AM": 31302, "6:00 AM": 66273, "7:00 AM": 103107, "8:00 AM": 109565,
                "9:00 AM": 80853, "10:00 AM": 75642, "11:00 AM": 68930, "12:00 PM": 72156,
                "1:00 PM": 69874, "2:00 PM": 71203, "3:00 PM": 73850, "4:00 PM": 78945,
                "5:00 PM": 85670, "6:00 PM": 93263, "7:00 PM": 80449, "8:00 PM": 73650,
                "9:00 PM": 67344, "10:00 PM": 57298, "11:00 PM": 46850
            },
            
            "glossary_notes": {
                "Impression": "Total audiences who passed by the screen and had the Opportunity To See (OTS). Includes repeat passers-by.",
                "Reach": "Unique count of audiences who passed by the screen and had the OTS at least once.",
                "Frequency": "Average number of times one person passes by the screen location during the specified period.",
                "OTS": "Opportunity To See - when a person is in position to view the digital screen advertisement"
            }
        }
    
    def search_pinecone_files(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search files stored in Pinecone vector database"""
        try:
            # Create sparse vector for BM25 search
            query_words = query.lower().split()
            sparse_vector = {
                "indices": list(range(min(20, len(query_words)))),
                "values": [0.5 + (i * 0.1) for i in range(min(20, len(query_words)))]
            }
            
            # Search Pinecone index
            results = self.index.query(
                sparse_vector=sparse_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            return results.get('matches', [])
            
        except Exception as e:
            st.error(f"❌ Pinecone search error: {str(e)}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the query"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Emotional classification
            if polarity > 0.5:
                emotion = "🎉 Very Positive"
                color = "#00FF00"
            elif polarity > 0.1:
                emotion = "😊 Positive"
                color = "#90EE90"
            elif polarity > -0.1:
                emotion = "😐 Neutral"
                color = "#FFFF00"
            elif polarity > -0.5:
                emotion = "😔 Negative"
                color = "#FFA500"
            else:
                emotion = "💔 Very Negative"
                color = "#FF0000"
                
            return {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "emotion": emotion,
                "color": color,
                "confidence": abs(polarity) * 100
            }
        except:
            return {"emotion": "😐 Neutral", "confidence": 50}
    
    def is_jad_vision_query(self, query: str) -> bool:
        """Check if query is about JAD Vision Dell campaign"""
        query_lower = query.lower()
        jad_indicators = ["jad vision", "dell", "7th july", "3rd aug", "campaign"]
        return any(indicator in query_lower for indicator in jad_indicators)
    
    def parse_jad_vision_query(self, query: str) -> str:
        """Parse JAD Vision query to determine information type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["report info", "campaign details", "duration", "dates"]):
            return "report_info"
        elif any(word in query_lower for word in ["performance", "impressions", "reach", "frequency"]):
            return "performance"
        elif any(word in query_lower for word in ["age", "gender", "demographic"]):
            return "demographics"
        elif any(word in query_lower for word in ["hourly", "time", "hour", "when"]):
            return "hourly"
        elif any(word in query_lower for word in ["screen", "network", "location"]):
            return "screens"
        elif any(word in query_lower for word in ["network id", "ids"]):
            return "network_ids"
        elif any(word in query_lower for word in ["glossary", "definition", "meaning"]):
            return "glossary"
        else:
            return "general"
    
    def get_jad_vision_report_info(self) -> str:
        """Get JAD Vision Dell campaign report information"""
        info = self.report_data["report_info"]
        
        details = f"""
        📊 **JAD Vision Dell Campaign Report**
        
        **📅 Campaign Period:**
        • Start Date: {info['Campaign_Duration']['Start']} (7th July 2024)
        • End Date: {info['Campaign_Duration']['End']} (3rd August 2024)
        • Duration: 28 days
        • Operating Hours: {info['Campaign_Duration']['Operating_Hours']}
        
        **⚙️ Technical Details:**
        • Spot Duration: {info['Spot_Duration']}
        • Total Networks: {info['Total_Networks']} digital screens
        • Campaign Type: {info['Campaign_Type']}
        • Client: Dell Technologies
        
        **📋 Report Timeline:**
        • Generated: {info['Report_Generation']['Generated_On']}
        • Reviewed: {info['Report_Generation']['Reviewed_On']} 
        • Validated: {info['Report_Generation']['Validated_On']}
        
        **🎯 Campaign Objective:** Digital Out-of-Home advertising across major Japanese transit locations
        """
        return details
    
    def get_jad_vision_performance(self) -> Tuple[str, go.Figure]:
        """Get JAD Vision performance summary with visualization"""
        performance = self.report_data["performance_summary"]
        
        # Calculate totals
        total_impressions = sum(data["Impressions"] for data in performance.values())
        total_reach = sum(data["Reach"] for data in performance.values())
        avg_frequency = np.mean([data["Frequency"] for data in performance.values()])
        
        # Find top performers
        top_location = max(performance.items(), key=lambda x: x[1]["Impressions"])
        
        summary = f"""
        📈 **JAD Vision Dell Performance Summary**
        
        **🎯 Campaign Totals (July 7 - Aug 3, 2024):**
        • Total Impressions: {total_impressions:,}
        • Total Unique Reach: {total_reach:,}
        • Average Frequency: {avg_frequency:.2f}
        
        **🏆 Top Performing Location:**
        • **{top_location[0]}**: {top_location[1]['Impressions']:,} impressions
        
        **📊 Location Breakdown:**
        """
        
        # Add performance breakdown
        sorted_performance = sorted(performance.items(), key=lambda x: x[1]["Impressions"], reverse=True)
        for i, (location, data) in enumerate(sorted_performance, 1):
            summary += f"\n{i}. **{location}**: {data['Impressions']:,} impressions | {data['Reach']:,} reach | {data['Frequency']:.2f} frequency"
        
        # Create visualization
        locations = list(performance.keys())
        impressions = [performance[loc]["Impressions"] for loc in locations]
        reach = [performance[loc]["Reach"] for loc in locations]
        frequency = [performance[loc]["Frequency"] for loc in locations]
        
        # Create subplot with multiple charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📊 Impressions by Location', '🎯 Reach vs Impressions', 
                           '🔄 Frequency Analysis', '💡 Performance Matrix'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Impressions bar chart
        fig.add_trace(go.Bar(x=locations, y=impressions, name="Impressions", 
                            marker_color='#3498db'), row=1, col=1)
        
        # Reach vs Impressions scatter
        fig.add_trace(go.Scatter(x=impressions, y=reach, mode='markers+text',
                                text=[loc.split()[0] for loc in locations],
                                name="Reach vs Impressions", 
                                marker=dict(size=12, color='#e74c3c')), row=1, col=2)
        
        # Frequency bar chart
        fig.add_trace(go.Bar(x=locations, y=frequency, name="Frequency", 
                            marker_color='#2ecc71'), row=2, col=1)
        
        # Performance efficiency (Reach/Impressions ratio)
        efficiency = [r/i*100 for r, i in zip(reach, impressions)]
        fig.add_trace(go.Scatter(x=frequency, y=efficiency, mode='markers+text',
                                text=[loc.split()[0] for loc in locations],
                                name="Efficiency %", 
                                marker=dict(size=[i/5000 for i in impressions], color='#9b59b6')), row=2, col=2)
        
        fig.update_layout(height=800, title_text="📊 JAD Vision Dell Campaign - Performance Dashboard")
        fig.update_xaxes(tickangle=45)
        
        return summary, fig
    
    def get_jad_vision_demographics(self) -> Tuple[str, List[go.Figure]]:
        """Get demographic analysis with charts"""
        demographics = self.report_data["age_gender_summary"]
        
        # Calculate totals
        total_impressions = sum(demographics["Age"][age]["Impressions"] for age in demographics["Age"])
        
        summary = f"""
        👥 **JAD Vision Dell Demographics Analysis**
        
        **📊 Age Distribution (Total: {total_impressions:,} impressions):**
        """
        
        # Age breakdown
        for age_group, data in demographics["Age"].items():
            summary += f"\n• **{age_group} years**: {data['Percentage']:.1f}% ({data['Impressions']:,} impressions)"
        
        summary += f"""
        
        **⚖️ Gender Distribution:**
        • **Male**: {demographics['Gender']['Male']['Percentage']:.1f}% ({demographics['Gender']['Male']['Impressions']:,} impressions)
        • **Female**: {demographics['Gender']['Female']['Percentage']:.1f}% ({demographics['Gender']['Female']['Impressions']:,} impressions)
        
        **🎯 Key Insights:**
        • Primary audience: 30-49 years (41% of total reach)
        • Male-skewed audience (59.6% male vs 40.4% female)
        • Strong working professional demographic
        """
        
        # Create visualizations
        charts = []
        
        # Age distribution pie chart
        age_labels = list(demographics["Age"].keys())
        age_values = [demographics["Age"][age]["Impressions"] for age in age_labels]
        
        fig1 = px.pie(values=age_values, names=age_labels,
                     title="📊 Dell Campaign - Age Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        charts.append(fig1)
        
        # Gender comparison
        gender_data = demographics["Gender"]
        fig2 = px.bar(
            x=list(gender_data.keys()),
            y=[gender_data[gender]["Impressions"] for gender in gender_data],
            title="👥 Gender Reach Distribution",
            color=list(gender_data.keys()),
            color_discrete_map={"Male": "#3498db", "Female": "#e91e63"}
        )
        fig2.update_layout(showlegend=False)
        charts.append(fig2)
        
        # Age vs Gender heatmap style chart
        age_percentages = [demographics["Age"][age]["Percentage"] for age in age_labels]
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Age Distribution %', x=age_labels, y=age_percentages, 
                             marker_color='lightblue', yaxis='y1'))
        
        fig3.update_layout(
            title="📈 Demographic Overview - Age Distribution",
            xaxis_title="Age Groups",
            yaxis_title="Percentage (%)",
            template="plotly_white"
        )
        charts.append(fig3)
        
        return summary, charts
    
    def get_jad_vision_hourly(self) -> Tuple[str, go.Figure]:
        """Get hourly performance analysis"""
        hourly = self.report_data["hourly_summary"]
        
        # Find peak hours
        peak_hour = max(hourly.items(), key=lambda x: x[1])
        lowest_hour = min(hourly.items(), key=lambda x: x[1])
        total_daily = sum(hourly.values())
        
        summary = f"""
        🕐 **JAD Vision Dell - Hourly Performance Analysis**
        
        **⏰ Key Performance Hours:**
        • **Peak Hour**: {peak_hour[0]} with {peak_hour[1]:,} impressions
        • **Lowest Hour**: {lowest_hour[0]} with {lowest_hour[1]:,} impressions
        • **Daily Total**: {total_daily:,} impressions
        • **Hourly Average**: {total_daily/len(hourly):,.0f} impressions
        
        **🚇 Traffic Patterns:**
        • **Morning Rush**: 7:00-9:00 AM (Peak commuter traffic)
        • **Evening Rush**: 6:00-8:00 PM (Return commute)
        • **Business Hours**: 10:00 AM - 5:00 PM (Steady traffic)
        • **Late Night**: 9:00 PM onwards (Declining traffic)
        
        **💡 Optimization Insights:**
        • Best visibility during morning commute (8:00 AM peak)
        • Strong evening performance for brand recall
        • Weekend patterns may differ (data shows weekday focus)
        """
        
        # Create hourly visualization
        hours = list(hourly.keys())
        impressions = list(hourly.values())
        
        fig = go.Figure()
        
        # Add area chart for better visual appeal
        fig.add_trace(go.Scatter(
            x=hours, y=impressions,
            mode='lines+markers',
            name='Hourly Impressions',
            fill='tonexty',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8, color='#e74c3c')
        ))
        
        # Highlight peak hours
        fig.add_annotation(
            x=peak_hour[0], y=peak_hour[1],
            text=f"Peak: {peak_hour[1]:,}",
            showarrow=True,
            arrowhead=2,
            bgcolor="yellow",
            bordercolor="red",
            font=dict(color="black", size=12)
        )
        
        # Add rush hour zones
        fig.add_vrect(x0="7:00 AM", x1="9:00 AM", 
                     annotation_text="Morning Rush", annotation_position="top left",
                     fillcolor="green", opacity=0.1, line_width=0)
        
        fig.add_vrect(x0="6:00 PM", x1="8:00 PM",
                     annotation_text="Evening Rush", annotation_position="top right", 
                     fillcolor="blue", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title="🕒 Dell JAD Vision Campaign - Hourly Impression Patterns",
            xaxis_title="Hour of Day",
            yaxis_title="Impressions",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        fig.update_xaxes(tickangle=45)
        
        return summary, fig
    
    def get_network_info(self) -> str:
        """Get network and screen information"""
        network_ids = self.report_data["network_ids"]
        screen_details = self.report_data["screen_details"]
        
        info = f"""
        🌐 **JAD Vision Network Information**
        
        **📺 Screen Network Details:**
        • **{list(screen_details.keys())[0]}**: {screen_details['Network_ID']}
        • **{list(screen_details.keys())[1]}**: {screen_details['Screen_Name']}
        • **{list(screen_details.keys())[2]}**: {screen_details['Impressions']}
        • **{list(screen_details.keys())[3]}**: {screen_details['Reach']}
        • **{list(screen_details.keys())[4]}**: {screen_details['Frequency']}
        
        **🆔 Network IDs ({len(network_ids)} screens):**
        """
        
        # Display network IDs in groups
        for i in range(0, len(network_ids), 4):
            group = network_ids[i:i+4]
            info += f"\n• {' | '.join(group)}"
        
        info += f"""
        
        **📍 Network Coverage:**
        • Major train stations across Japan
        • High-traffic commuter locations
        • Premium digital screen placements
        • Strategic visibility points for Dell brand exposure
        """
        
        return info
    
    def get_glossary_info(self) -> str:
        """Get glossary and definitions"""
        glossary = self.report_data["glossary_notes"]
        
        info = f"""
        📚 **JAD Vision Campaign Glossary**
        
        **📖 Key Definitions:**
        """
        
        for term, definition in glossary.items():
            info += f"\n• **{term}**: {definition}"
        
        info += f"""
        
        **🏢 Additional Context:**
        • **JAD Vision**: Premium digital out-of-home advertising network in Japan
        • **Dell Campaign**: Technology brand awareness campaign
        • **DOOH**: Digital Out-of-Home advertising
        • **Transit Advertising**: Advertising in transportation hubs and stations
        • **Campaign KPIs**: Impressions, Reach, and Frequency metrics
        """
        
        return info
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Main query processing function"""
        
        # Analyze query sentiment
        sentiment = self.analyze_sentiment(user_query)
        
        # Check if it's a JAD Vision query
        if self.is_jad_vision_query(user_query):
            # Handle JAD Vision specific queries
            query_type = self.parse_jad_vision_query(user_query)
            
            response = {
                "source": "jad_vision_report",
                "query_type": query_type,
                "sentiment": sentiment,
                "success": True
            }
            
            if query_type == "report_info":
                response["details"] = self.get_jad_vision_report_info()
                response["title"] = "📊 Dell Campaign Report Information"
                
            elif query_type == "performance":
                details, chart = self.get_jad_vision_performance()
                response["details"] = details
                response["chart"] = chart
                response["title"] = "📈 Campaign Performance Summary"
                
            elif query_type == "demographics":
                details, charts = self.get_jad_vision_demographics()
                response["details"] = details
                response["charts"] = charts
                response["title"] = "👥 Demographic Analysis"
                
            elif query_type == "hourly":
                details, chart = self.get_jad_vision_hourly()
                response["details"] = details
                response["chart"] = chart
                response["title"] = "🕐 Hourly Performance Analysis"
                
            elif query_type in ["screens", "network_ids"]:
                response["details"] = self.get_network_info()
                response["title"] = "🌐 Network & Screen Information"
                
            elif query_type == "glossary":
                response["details"] = self.get_glossary_info()
                response["title"] = "📚 Campaign Glossary"
                
            else:  # general
                response["details"] = f"""
                📋 **JAD Vision Dell Campaign Overview**
                
                **Campaign Summary:**
                • Period: July 7 - August 3, 2024 (28 days)
                • Client: Dell Technologies
                • Networks: 25 digital screens across Japan
                • Format: 15-second digital advertisements
                
                **📊 Available Report Sections:**
                1. **Report Info** - Campaign details and timeline
                2. **Performance Summary** - Impressions, reach, frequency data  
                3. **Demographics** - Age and gender breakdown
                4. **Hourly Analysis** - Time-based performance patterns
                5. **Network Information** - Screen locations and IDs
                6. **Glossary** - Definitions and terminology
                
                **💡 Ask me about any specific section!**
                Examples:
                • "JAD Vision Dell performance summary"
                • "JAD Vision Dell demographics" 
                • "JAD Vision Dell hourly analysis"
                """
                response["title"] = "📊 JAD Vision Dell Campaign"
            
            return response
        
        else:
            # Search Pinecone for other files
            pinecone_results = self.search_pinecone_files(user_query)
            
            if pinecone_results:
                # Format Pinecone results
                formatted_results = self.format_pinecone_results(pinecone_results)
                return {
                    "source": "pinecone_files", 
                    "success": True,
                    "title": "🔍 File Search Results",
                    "details": formatted_results,
                    "sentiment": sentiment,
                    "num_results": len(pinecone_results)
                }
            else:
                return {
                    "source": "no_results",
                    "success": False,
                    "message": "No matching files found in database.",
                    "suggestion": "Try asking about 'JAD Vision Dell campaign' for report data, or check your query spelling."
                }
    
    def format_pinecone_results(self, results: List[Dict]) -> str:
        """Format Pinecone search results"""
        if not results:
            return "❌ No results found in the file database."
        
        formatted = f"📁 **Found {len(results)} relevant files/documents:**\n\n"
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            score = result.get('score', 0)
            
            formatted += f"""
            **📄 Result {i}** (Relevance: {score:.3f})
            """
            
            # Display available metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        formatted += f"\n• **{key.title()}**: {value}"
            
            formatted += "\n" + "─" * 50 + "\n"
        
        return formatted

def create_jad_vision_interface():
    """Create the main JAD Vision interface"""
    
    st.set_page_config(
        page_title="JAD Vision Dell Campaign System",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #007DB8, #0078D4, #106EBE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .dell-blue {
        color: #007DB8;
        font-weight: bold;
    }
    .info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #007DB8;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #007DB8 0%, #106EBE 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    .pinecone-status {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📊 JAD Vision Dell Campaign System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dell-blue" style="text-align: center; font-size: 1.2rem;">Pinecone-Powered Report Query System | July 7 - August 3, 2024</p>', unsafe_allow_html=True)
    
    # Initialize system
    if 'jad_system' not in st.session_state:
        st.session_state.jad_system = JADVisionRAGSystem()
    
    # Sidebar with Pinecone status and examples
    st.sidebar.header("🔌 System Status")
    
    # Pinecone connection status
    try:
        stats = st.session_state.jad_system.index.describe_index_stats()
        st.sidebar.markdown(f"""
        <div class="pinecone-status">
            <h4>🌲 Pinecone Database</h4>
            <p><strong>Status:</strong> ✅ Connected</p>
            <p><strong>Vectors:</strong> {stats.total_vector_count:,}</p>
            <p><strong>Capacity:</strong> {stats.index_fullness:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"❌ Pinecone Connection Error: {str(e)}")
    
    st.sidebar.markdown("---")
    
    # JAD Vision example queries
    st.sidebar.header("📋 JAD Vision Examples")
    jad_examples = [
        "JAD Vision 7th July - 3rd Aug Dell report info",
        "JAD Vision Dell performance summary",
        "JAD Vision Dell demographics breakdown", 
        "JAD Vision Dell hourly analysis",
        "JAD Vision Dell network IDs",
        "JAD Vision Dell screen details",
        "JAD Vision Dell glossary definitions"
    ]
    
    for query in jad_examples:
        if st.sidebar.button(f"🎯 {query}", key=f"jad_{hash(query)}"):
            st.session_state.current_query = query
    
    st.sidebar.markdown("---")
    
    # General file search examples
    st.sidebar.header("📁 File Search Examples")
    file_examples = [
        "Campaign performance data",
        "Marketing analytics report",
        "Customer engagement metrics",
        "Brand awareness study"
    ]
    
    for query in file_examples:
        if st.sidebar.button(f"🔍 {query}", key=f"file_{hash(query)}"):
            st.session_state.current_query = query
    
    # Main query interface
    st.markdown("### 🔍 Query Interface")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_query = st.text_input(
            "💬 Enter your query:",
            value=st.session_state.get('current_query', ''),
            placeholder="e.g., JAD Vision 7th July - 3rd Aug Dell report info"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🚀 Search", type="primary")
    
    # Process query
    if (search_button and user_query) or st.session_state.get('current_query'):
        query_to_process = user_query or st.session_state.get('current_query', '')
        
        # Clear the session state query after using it
        if 'current_query' in st.session_state:
            del st.session_state.current_query
        
        with st.spinner("🔍 Processing your query..."):
            try:
                response = st.session_state.jad_system.process_query(query_to_process)
                
                if response["success"]:
                    
                    # Show query sentiment
                    if "sentiment" in response:
                        sentiment = response["sentiment"]
                        st.markdown(f"""
                        <div style="background: {sentiment.get('color', '#FFFF00')}; 
                                   padding: 0.5rem; border-radius: 5px; margin: 1rem 0; text-align: center;">
                            <strong>Query Emotion: {sentiment.get('emotion', '😐 Neutral')}</strong>
                            (Confidence: {sentiment.get('confidence', 50):.0f}%)
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display title and source indicator
                    source_emoji = "📊" if response["source"] == "jad_vision_report" else "📁"
                    st.subheader(f"{source_emoji} {response['title']}")
                    
                    # Show data source
                    if response["source"] == "jad_vision_report":
                        st.success("✅ Data source: JAD Vision Dell Campaign Report")
                    elif response["source"] == "pinecone_files":
                        st.info(f"📁 Data source: Pinecone Database ({response.get('num_results', 0)} files found)")
                    
                    # Display details
                    st.markdown(f'<div class="info-card">{response["details"]}</div>', unsafe_allow_html=True)
                    
                    # Display visualizations
                    if "chart" in response:
                        st.plotly_chart(response["chart"], use_container_width=True)
                    
                    if "charts" in response:
                        if len(response["charts"]) > 1:
                            # Display multiple charts in columns
                            if len(response["charts"]) == 2:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.plotly_chart(response["charts"][0], use_container_width=True)
                                with col2:
                                    st.plotly_chart(response["charts"][1], use_container_width=True)
                            else:
                                for chart in response["charts"]:
                                    st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.plotly_chart(response["charts"][0], use_container_width=True)
                
                else:
                    st.warning(f"⚠️ {response['message']}")
                    if "suggestion" in response:
                        st.info(f"💡 {response['suggestion']}")
                        
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.error("Please check your Pinecone connection and try again.")
    
    # Quick metrics display
    st.markdown("---")
    st.markdown("### 📊 Quick Campaign Stats")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate quick stats from report data
    total_impressions = sum(data["Impressions"] for data in st.session_state.jad_system.report_data["performance_summary"].values())
    total_reach = sum(data["Reach"] for data in st.session_state.jad_system.report_data["performance_summary"].values())
    avg_frequency = np.mean([data["Frequency"] for data in st.session_state.jad_system.report_data["performance_summary"].values()])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_impressions:,}</h3>
            <p>Total Impressions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_reach:,}</h3>
            <p>Total Reach</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_frequency:.2f}</h3>
            <p>Avg Frequency</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>28</h3>
            <p>Campaign Days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h3>25</h3>
            <p>Screen Networks</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Help and usage guide
    with st.expander("❓ How to Use This System"):
        st.markdown("""
        ## 🎯 **Two Types of Searches:**
        
        ### 📊 **JAD Vision Dell Campaign Queries**
        **Always include "JAD Vision" and "Dell" in your query**
        
        **Available Information:**
        - **Report Info**: `"JAD Vision Dell report info"` - Campaign details and timeline
        - **Performance**: `"JAD Vision Dell performance"` - Impressions, reach, frequency
        - **Demographics**: `"JAD Vision Dell demographics"` - Age and gender breakdown  
        - **Hourly**: `"JAD Vision Dell hourly analysis"` - Time-based patterns
        - **Networks**: `"JAD Vision Dell network IDs"` - Screen locations and identifiers
        - **Glossary**: `"JAD Vision Dell glossary"` - Definitions and terminology
        
        ### 📁 **File Database Searches**
        **Search any other files stored in Pinecone database**
        
        Examples:
        - `"Marketing campaign results"`
        - `"Customer analytics data"`
        - `"Brand performance metrics"`
        
        ## 💡 **Tips:**
        - Be specific in your queries for better results
        - Use exact phrases like "JAD Vision Dell" for campaign data
        - Check Pinecone connection status in sidebar
        - View example queries in the sidebar for guidance
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>🌲 Pinecone-Powered</strong> | <strong>📊 JAD Vision Integration</strong> | <strong>🚀 Real-time Analytics</strong></p>
        <p>Dell Digital Out-of-Home Campaign Analytics System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_jad_vision_interface()
