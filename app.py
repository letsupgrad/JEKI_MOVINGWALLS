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

# Query for Screen Details
def get_screen_details():
    screen_details = {
        "Network_ID": "Unique identifier for each screen in the network",
        "Screen_Name": "Name of each screen in the network",
        "Impressions": "Total number of people who passed by the screen and had the Opportunity To See (OTS)",
        "Reach": "Unique count of audiences who passed by the screen and had the OTS at least once",
        "Frequency": "Average number of times a person is exposed to an ad during the specified period"
    }
    return screen_details

# Query for Network ID
def get_network_ids():
    network_ids = [
        "JPN-JEK-D-00000-00029", "JPN-JEK-D-00000-00030", "JPN-JEK-D-00000-00031",
        "JPN-JEK-D-00000-00032", "JPN-JEK-D-00000-00033", "JPN-JEK-D-00000-00034",
        "JPN-JEK-D-00000-00035", "JPN-JEK-D-00000-00036", "JPN-JEK-D-00000-00039",
        "JPN-JEK-D-00000-00040", "JPN-JEK-D-00000-00041", "JPN-JEK-D-00000-00042",
        "JPN-JEK-D-00000-00044", "JPN-JEK-D-00000-00045", "JPN-JEK-D-00000-00046",
        "JPN-JEK-D-00000-00047", "JPN-JEK-D-00000-00048", "JPN-JEK-D-00000-00049",
        "JPN-JEK-D-00000-00050", "JPN-JEK-D-00000-00051", "JPN-JEK-D-00000-00052",
        "JPN-JEK-D-00000-00058", "JPN-JEK-D-00000-00059", "JPN-JEK-D-00000-00060",
        "JPN-JEK-D-00000-00061"
    ]
    return network_ids

# Query for Overall Performance Summary
def get_overall_performance_summary():
    performance_summary = {
        "Sugamo Station Exit": {"Impressions": 49682, "Frequency": 1.66, "Reach": 29929},
        "Shinjuku Station East Exit": {"Impressions": 61259, "Frequency": 1.11, "Reach": 55188},
        "Shinjuku Station South Exit": {"Impressions": 63881, "Frequency": 1.00, "Reach": 63881},
        "Shinjuku Station Koshu-Kaido Exit": {"Impressions": 50093, "Frequency": 1.17, "Reach": 42815},
        "Shibuya Station Hachiko Exit": {"Impressions": 53843, "Frequency": 1.22, "Reach": 44134},
        "Gotanda Station": {"Impressions": 21828, "Frequency": 1.46, "Reach": 14951},
        "Shinagawa Station Central Gate": {"Impressions": 9342, "Frequency": 1.05, "Reach": 8897},
        "Takanawa Gateway Station": {"Impressions": 3669, "Frequency": 1.60, "Reach": 2293},
        "Yurakucho Station Central Gate": {"Impressions": 13464, "Frequency": 1.16, "Reach": 11607},
        "Tokyo Station Marunouchi Underground Passage": {"Impressions": 19369, "Frequency": 1.23, "Reach": 15747},
        "Tokyo Station Keiyo Passage": {"Impressions": 17793, "Frequency": 1.20, "Reach": 14828},
        "Akihabara Station New Electric Town Exit": {"Impressions": 50248, "Frequency": 1.13, "Reach": 44467},
        "Kichijoji Station North-South Free Passage": {"Impressions": 14756, "Frequency": 1.22, "Reach": 12095},
        "Urawa Station Gate": {"Impressions": 53167, "Frequency": 1.35, "Reach": 39383},
        "Omiya Station Central Gate": {"Impressions": 43421, "Frequency": 1.42, "Reach": 30578},
        "Yokohama Station Central Passage": {"Impressions": 103351, "Frequency": 1.16, "Reach": 89096},
        "JR Yokohama Tower Atrium": {"Impressions": 27946, "Frequency": 1.06, "Reach": 26364},
        "Takadanobaba Station Smile Vision": {"Impressions": 51685, "Frequency": 1.35, "Reach": 38285},
        "Ikebukuro Station Central Gate": {"Impressions": 107227, "Frequency": 1.19, "Reach": 90107},
        "Sakuragicho Station": {"Impressions": 32180, "Frequency": 1.38, "Reach": 23319},
        "Yokohama Station South Gate": {"Impressions": 69742, "Frequency": 1.43, "Reach": 48771},
        "Tokyo Station Shinkansen North Transfer Gate": {"Impressions": 4725, "Frequency": 1.00, "Reach": 4725},
        "Tokyo Station Shinkansen South Transfer Gate": {"Impressions": 4514, "Frequency": 1.00, "Reach": 4514},
        "Ebisu Station West Exit": {"Impressions": 62157, "Frequency": 1.04, "Reach": 59766},
        "Akabane Station North Gate": {"Impressions": 31991, "Frequency": 1.01, "Reach": 31674}
    }
    return performance_summary


# Query for Overall Age and Gender
def get_overall_age_gender():
    age_gender_summary = {
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
    }
    return age_gender_summary

# Query for Overall Hourly
def get_overall_hourly():
    hourly_summary = {
        "5:00 AM": 31302,
        "6:00 AM": 66273,
        "7:00 AM": 103107,
        "8:00 AM": 109565,
        "9:00 AM": 80853,
        "10:00 AM": 75642,
        "11:00 AM": 68930,
        "12:00 PM": 72156,
        "1:00 PM": 69874,
        "2:00 PM": 71203,
        "3:00 PM": 73850,
        "4:00 PM": 78945,
        "5:00 PM": 85670,
        "6:00 PM": 93263,
        "7:00 PM": 80449,
        "8:00 PM": 73650,
        "9:00 PM": 67344,
        "10:00 PM": 57298,
        "11:00 PM": 46850
    }
    return hourly_summary

# Query for Report Info
def get_report_info():
    report_info = {
        "Campaign_Duration": {
            "Start": "2024-03-04",
            "End": "2024-03-10",
            "Operating_Hours": {
                "March 4-8": "Morning and Evening",
                "March 9-10": "Afternoon"
            }
        },
        "Spot_Duration": "15 seconds",
        "Report_Generation": {
            "Generated_On": "2024-04-04",
            "Reviewed_On": "2024-04-09",
            "Validated_On": "2024-04-09"
        },
        "Report_Structure": [
            "Screen Details",
            "Overall Performance Summary",
            "Daily Summary",
            "Overall Age and Gender",
            "Overall Hourly",
            "Network Summary"
        ]
    }
    return report_info

# Query for Glossary & Notes
def get_glossary_notes():
    glossary_notes = {
        "Glossary": {
            "Impression": "Total audiences who passed by the screen and had the Opportunity To See (OTS). Includes repeat passers-by.",
            "Reach": "Unique count of audiences who passed by the screen and had the OTS at least once.",
            "Frequency": "Average number of times one person passes by the screen location during the specified period."
        },
        "Notes": {
            "Billboard_Details_Tab": [
                "Overall Performance Summary",
                "Weekly Summary",
                "Daily Summary",
                "Overall Age and Gender",
                "Overall Hourly",
                "Network Summary"
            ]
        }
    }
    return glossary_notes
   
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
            "report_info": get_report_info(),  # Use the defined function
            "screen_details": get_screen_details(), # Use the defined function
            "network_ids": get_network_ids(),      # Use the defined function
            "performance_summary": get_overall_performance_summary(), # Use the defined function
            "age_gender_summary": get_overall_age_gender(), # Use the defined function
            "hourly_summary": get_overall_hourly(), # Use the defined function
            "glossary_notes": get_glossary_notes() # Use the defined function
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
        # Updated indicators based on the provided data for March 4-10, 2024
        jad_indicators = ["jad vision", "march 4", "march 10", "campaign"]
        return any(indicator in query_lower for indicator in jad_indicators)
    
    def parse_jad_vision_query(self, query: str) -> str:
        """Parse JAD Vision query to determine information type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["report info", "campaign details", "duration", "dates", "report structure", "generated on", "reviewed on", "validated on"]):
            return "report_info"
        elif any(word in query_lower for word in ["performance", "impressions", "reach", "frequency", "overall performance summary"]):
            return "performance"
        elif any(word in query_lower for word in ["age", "gender", "demographic", "overall age and gender"]):
            return "demographics"
        elif any(word in query_lower for word in ["hourly", "time", "hour", "when", "overall hourly"]):
            return "hourly"
        elif any(word in query_lower for word in ["screen", "network", "location", "screen details", "total referenceid"]):
            return "screens"
        elif any(word in query_lower for word in ["network id", "ids"]):
            return "network_ids"
        elif any(word in query_lower for word in ["glossary", "definition", "meaning", "notes", "glossary & note"]):
            return "glossary"
        else:
            return "general"
    
    def get_jad_vision_report_info(self) -> str:
        """Get JAD Vision Dell campaign report information"""
        info = self.report_data["report_info"]
        
        # Safely access nested dictionary values
        campaign_name = info.get("Campaign_Duration", {}).get("Campaign_Name", "JAD Vision Campaign")
        start_date = info.get("Campaign_Duration", {}).get("Start", "N/A")
        end_date = info.get("Campaign_Duration", {}).get("End", "N/A")
        
        # Calculate duration if dates are available and valid
        duration_str = "N/A"
        try:
            if start_date != "N/A" and end_date != "N/A":
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                duration_days = (end_dt - start_dt).days + 1 # Include both start and end day
                duration_str = f"{duration_days} days"
        except ValueError:
            pass # Keep duration_str as N/A if date parsing fails

        operating_hours_march4_8 = info.get("Campaign_Duration", {}).get("Operating_Hours", {}).get("March 4-8", "N/A")
        operating_hours_march9_10 = info.get("Campaign_Duration", {}).get("Operating_Hours", {}).get("March 9-10", "N/A")
        
        spot_duration = info.get("Spot_Duration", "N/A")
        
        generated_on = info.get("Report_Generation", {}).get("Generated_On", "N/A")
        reviewed_on = info.get("Report_Generation", {}).get("Reviewed_On", "N/A")
        validated_on = info.get("Report_Generation", {}).get("Validated_On", "N/A")

        report_structure = "\n".join([f"• {item}" for item in info.get("Report_Structure", [])])
        
        details = f"""
        📊 **{campaign_name} Report**
        
        **📅 Campaign Period:**
        • Start Date: {start_date}
        • End Date: {end_date}
        • Duration: {duration_str}
        • Operating Hours (March 4-8): {operating_hours_march4_8}
        • Operating Hours (March 9-10): {operating_hours_march9_10}
        
        **⚙️ Technical Details:**
        • Spot Duration: {spot_duration}
        
        **📋 Report Timeline:**
        • Generated: {generated_on}
        • Reviewed: {reviewed_on} 
        • Validated: {validated_on}
        
        **📝 Report Structure:**
        {report_structure}
        """
        return details
    
    def get_jad_vision_performance(self) -> Tuple[str, go.Figure]:
        """Get JAD Vision performance summary with visualization"""
        performance = self.report_data["performance_summary"]
        
        # Calculate totals
        total_impressions = sum(data["Impressions"] for data in performance.values())
        total_reach = sum(data["Reach"] for data in performance.values())
        
        # Calculate average frequency carefully to avoid division by zero
        frequencies = [data["Frequency"] for data in performance.values() if data["Frequency"] is not None]
        avg_frequency = np.mean(frequencies) if frequencies else 0.0
        
        # Find top performers
        top_location = max(performance.items(), key=lambda x: x[1]["Impressions"])
        
        summary = f"""
        📈 **JAD Vision Campaign Performance Summary**
        
        **🎯 Campaign Totals (March 4 - March 10, 2024):**
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
        efficiency = []
        for r, i in zip(reach, impressions):
            if i > 0: # Avoid division by zero
                efficiency.append(r/i*100)
            else:
                efficiency.append(0) # Or another suitable default value
        
        fig.add_trace(go.Scatter(x=frequency, y=efficiency, mode='markers+text',
                                text=[loc.split()[0] for loc in locations],
                                name="Efficiency %", 
                                marker=dict(size=[i/5000 for i in impressions], color='#9b59b6')), row=2, col=2)
        
        fig.update_layout(height=800, title_text="📊 JAD Vision Campaign - Performance Dashboard")
        fig.update_xaxes(tickangle=45)
        
        return summary, fig
    
    def get_jad_vision_demographics(self) -> Tuple[str, List[go.Figure]]:
        """Get demographic analysis with charts"""
        demographics = self.report_data["age_gender_summary"]
        
        # Calculate totals
        total_impressions = sum(demographics["Age"][age]["Impressions"] for age in demographics["Age"])
        
        summary = f"""
        👥 **JAD Vision Demographics Analysis**
        
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
        • Primary audience: 40-49 years (22.5% of total impressions)
        • Male-skewed audience (59.6% male vs 40.4% female)
        """
        
        # Create visualizations
        charts = []
        
        # Age distribution pie chart
        age_labels = list(demographics["Age"].keys())
        age_values = [demographics["Age"][age]["Impressions"] for age in age_labels]
        
        fig1 = px.pie(values=age_values, names=age_labels,
                     title="📊 JAD Vision Campaign - Age Distribution",
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
        🕐 **JAD Vision - Hourly Performance Analysis**
        
        **⏰ Key Performance Hours:**
        • **Peak Hour**: {peak_hour[0]} with {peak_hour[1]:,} impressions
        • **Lowest Hour**: {lowest_hour[0]} with {lowest_hour[1]:,} impressions
        • **Daily Total**: {total_daily:,} impressions (assuming this is a typical daily sum)
        • **Hourly Average**: {total_daily/len(hourly):,.0f} impressions
        
        **🚇 Traffic Patterns:**
        • **Morning Rush**: 7:00-9:00 AM (Peak commuter traffic)
        • **Evening Rush**: 6:00-8:00 PM (Return commute)
        • **Business Hours**: 10:00 AM - 5:00 PM (Steady traffic)
        • **Late Night**: 9:00 PM onwards (Declining traffic)
        
        **💡 Optimization Insights:**
        • Best visibility during morning commute (8:00 AM peak)
        • Strong evening performance for brand recall
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
        
        # Add rush hour zones (adjusting for available data)
        # Assuming peak hours 7-9 AM and 6-8 PM are "rush hours" from the context
        morning_rush_start_idx = hours.index("7:00 AM") if "7:00 AM" in hours else None
        morning_rush_end_idx = hours.index("9:00 AM") if "9:00 AM" in hours else None

        evening_rush_start_idx = hours.index("6:00 PM") if "6:00 PM" in hours else None
        evening_rush_end_idx = hours.index("8:00 PM") if "8:00 PM" in hours else None
        
        if morning_rush_start_idx is not None and morning_rush_end_idx is not None:
            fig.add_vrect(x0=hours[morning_rush_start_idx], x1=hours[morning_rush_end_idx],
                         annotation_text="Morning Rush", annotation_position="top left",
                         fillcolor="green", opacity=0.1, line_width=0)
        
        if evening_rush_start_idx is not None and evening_rush_end_idx is not None:
            fig.add_vrect(x0=hours[evening_rush_start_idx], x1=hours[evening_rush_end_idx],
                         annotation_text="Evening Rush", annotation_position="top right", 
                         fillcolor="blue", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title="🕒 JAD Vision Campaign - Hourly Impression Patterns",
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
        
        # Get overall performance summary to list impressions per screen
        performance_summary = self.report_data["performance_summary"]
        
        info = f"""
        🌐 **JAD Vision Network Information**
        
        **📺 Screen Network Details:**
        • **Network ID**: {screen_details['Network_ID']}
        • **Screen Name**: {screen_details['Screen_Name']}
        • **Impressions**: {screen_details['Impressions']}
        • **Reach**: {screen_details['Reach']}
        • **Frequency**: {screen_details['Frequency']}
        
        **🆔 Network IDs ({len(network_ids)} screens):**
        """
        
        # Display network IDs and their performance
        for i, network_id in enumerate(network_ids, 1):
            # Try to find the screen name corresponding to the network_id
            screen_name = "Unknown Location"
            for name, data in performance_summary.items():
                # This requires a mapping between network_id and screen name if not directly available
                # For now, we'll just use the network_id and try to get performance data
                if name.replace(" ", "") in network_id.replace(" ", ""): # Simple heuristic
                    screen_name = name
                    break
            
            perf_data = performance_summary.get(screen_name, {"Impressions": "N/A", "Reach": "N/A", "Frequency": "N/A"})
            
            info += f"\n{i}. **{network_id}** ({screen_name}): Impressions: {perf_data['Impressions']:,}, Reach: {perf_data['Reach']:,}, Frequency: {perf_data['Frequency']:.2f}"
        
        info += f"""
        
        **📍 Network Coverage:**
        • Major train stations across Japan (as indicated by the specific station names)
        • High-traffic commuter locations
        • Premium digital screen placements
        • Strategic visibility points for brand exposure
        """
        
