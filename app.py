import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import re
from datetime import datetime

class JADVisionReportSystem:
    def __init__(self):
        """Initialize the JAD Vision Report Query System"""
        self.report_data = self.load_report_data()
        
    def load_report_data(self) -> Dict[str, Any]:
        """Load all JAD Vision report data"""
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
    
    def parse_query(self, user_query: str) -> Dict[str, Any]:
        """Parse user query and determine what information to show"""
        query_lower = user_query.lower()
        
        # Check if it's about JAD Vision Dell report
        if "jad vision" in query_lower and "dell" in query_lower:
            
            # Determine what specific information they want
            query_type = "general"
            
            if any(word in query_lower for word in ["report info", "campaign details", "duration", "dates"]):
                query_type = "report_info"
            elif any(word in query_lower for word in ["screen", "network", "location"]):
                query_type = "screens"
            elif any(word in query_lower for word in ["performance", "impressions", "reach", "frequency"]):
                query_type = "performance"
            elif any(word in query_lower for word in ["age", "gender", "demographic"]):
                query_type = "demographics"
            elif any(word in query_lower for word in ["hourly", "time", "hour", "when"]):
                query_type = "hourly"
            elif any(word in query_lower for word in ["glossary", "definition", "meaning"]):
                query_type = "glossary"
            elif any(word in query_lower for word in ["network id", "ids"]):
                query_type = "network_ids"
            
            return {
                "is_jad_vision_query": True,
                "query_type": query_type,
                "original_query": user_query
            }
        
        return {
            "is_jad_vision_query": False,
            "message": "Please ask about 'JAD Vision Dell' campaign to get report details."
        }
    
    def get_report_info_details(self) -> str:
        """Get detailed report information"""
        info = self.report_data["report_info"]
        
        details = f"""
        📊 **JAD Vision Dell Campaign Report**
        
        **Campaign Period:**
        • Start Date: {info['Campaign_Duration']['Start']}
        • End Date: {info['Campaign_Duration']['End']}
        • Duration: 28 days
        • Operating Hours: {info['Campaign_Duration']['Operating_Hours']}
        
        **Technical Details:**
        • Spot Duration: {info['Spot_Duration']}
        • Total Networks: {info['Total_Networks']} screens
        • Campaign Type: {info['Campaign_Type']}
        
        **Report Timeline:**
        • Generated: {info['Report_Generation']['Generated_On']}
        • Reviewed: {info['Report_Generation']['Reviewed_On']}
        • Validated: {info['Report_Generation']['Validated_On']}
        """
        return details
    
    def get_screen_details_info(self) -> str:
        """Get screen details information"""
        details = self.report_data["screen_details"]
        
        info = f"""
        🖥️ **Screen Network Details**
        
        **Data Fields Available:**
        • **{list(details.keys())[0]}**: {details['Network_ID']}
        • **{list(details.keys())[1]}**: {details['Screen_Name']}
        • **{list(details.keys())[2]}**: {details['Impressions']}
        • **{list(details.keys())[3]}**: {details['Reach']}
        • **{list(details.keys())[4]}**: {details['Frequency']}
        
        **Total Networks in Campaign:** {len(self.report_data['network_ids'])} screens across Japan
        """
        return info
    
    def get_performance_summary(self) -> Tuple[str, go.Figure]:
        """Get performance summary with visualization"""
        performance = self.report_data["performance_summary"]
        
        # Create summary text
        total_impressions = sum(data["Impressions"] for data in performance.values())
        total_reach = sum(data["Reach"] for data in performance.values())
        avg_frequency = np.mean([data["Frequency"] for data in performance.values()])
        
        summary = f"""
        📈 **Overall Performance Summary**
        
        **Campaign Totals:**
        • Total Impressions: {total_impressions:,}
        • Total Reach: {total_reach:,}
        • Average Frequency: {avg_frequency:.2f}
        
        **Top Performing Locations:**
        """
        
        # Add top performers
        sorted_performance = sorted(performance.items(), key=lambda x: x[1]["Impressions"], reverse=True)
        for i, (location, data) in enumerate(sorted_performance[:5], 1):
            summary += f"\n{i}. **{location}**: {data['Impressions']:,} impressions, {data['Reach']:,} reach"
        
        # Create visualization
        locations = list(performance.keys())
        impressions = [performance[loc]["Impressions"] for loc in locations]
        reach = [performance[loc]["Reach"] for loc in locations]
        frequency = [performance[loc]["Frequency"] for loc in locations]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Impressions by Location', 'Reach vs Impressions', 'Frequency Analysis', 'Performance Matrix'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Impressions bar chart
        fig.add_trace(go.Bar(x=locations, y=impressions, name="Impressions", marker_color='lightblue'), row=1, col=1)
        
        # Reach vs Impressions scatter
        fig.add_trace(go.Scatter(x=impressions, y=reach, mode='markers+text', 
                                text=[loc[:10] + "..." if len(loc) > 10 else loc for loc in locations],
                                name="Reach vs Impressions", marker=dict(size=10, color='orange')), row=1, col=2)
        
        # Frequency bar chart
        fig.add_trace(go.Bar(x=locations, y=frequency, name="Frequency", marker_color='lightgreen'), row=2, col=1)
        
        # Performance matrix
        fig.add_trace(go.Scatter(x=frequency, y=impressions, mode='markers+text',
                                text=[loc[:8] + "..." if len(loc) > 8 else loc for loc in locations],
                                name="Performance Matrix", 
                                marker=dict(size=[r/5000 for r in reach], color='purple')), row=2, col=2)
        
        fig.update_layout(height=800, title_text="📊 JAD Vision Dell Campaign Performance Dashboard")
        fig.update_xaxes(tickangle=45)
        
        return summary, fig
    
    def get_demographics_info(self) -> Tuple[str, List[go.Figure]]:
        """Get demographic breakdown with visualizations"""
        demographics = self.report_data["age_gender_summary"]
        
        # Summary text
        summary = f"""
        👥 **Demographic Analysis**
        
        **Age Distribution:**
        """
        
        # Age breakdown
        total_age_impressions = sum(demographics["Age"][age]["Impressions"] for age in demographics["Age"])
        for age_group, data in demographics["Age"].items():
            summary += f"\n• **{age_group}**: {data['Percentage']:.1f}% ({data['Impressions']:,} impressions)"
        
        summary += f"""
        
        **Gender Distribution:**
        • **Male**: {demographics['Gender']['Male']['Percentage']:.1f}% ({demographics['Gender']['Male']['Impressions']:,} impressions)
        • **Female**: {demographics['Gender']['Female']['Percentage']:.1f}% ({demographics['Gender']['Female']['Impressions']:,} impressions)
        """
        
        # Create visualizations
        charts = []
        
        # Age distribution pie chart
        age_labels = list(demographics["Age"].keys())
        age_values = [demographics["Age"][age]["Impressions"] for age in age_labels]
        
        fig1 = px.pie(values=age_values, names=age_labels, 
                     title="📊 Age Distribution - Dell Campaign Reach",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        charts.append(fig1)
        
        # Gender distribution bar chart
        gender_data = demographics["Gender"]
        fig2 = px.bar(
            x=list(gender_data.keys()), 
            y=[gender_data[gender]["Impressions"] for gender in gender_data],
            title="👥 Gender Reach Distribution",
            color=list(gender_data.keys()),
            color_discrete_map={"Male": "#3498db", "Female": "#e74c3c"}
        )
        charts.append(fig2)
        
        # Combined demographic heatmap
        age_percentages = [demographics["Age"][age]["Percentage"] for age in age_labels]
        gender_percentages = [demographics["Gender"][gender]["Percentage"] for gender in gender_data]
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Age Groups', x=age_labels, y=age_percentages, marker_color='lightblue'))
        fig3.add_trace(go.Bar(name='Gender', x=list(gender_data.keys()), y=gender_percentages, marker_color='lightcoral'))
        fig3.update_layout(title="📈 Demographic Comparison Overview", barmode='group')
        charts.append(fig3)
        
        return summary, charts
    
    def get_hourly_analysis(self) -> Tuple[str, go.Figure]:
        """Get hourly performance analysis"""
        hourly = self.report_data["hourly_summary"]
        
        # Find peak hours
        peak_hour = max(hourly.items(), key=lambda x: x[1])
        lowest_hour = min(hourly.items(), key=lambda x: x[1])
        total_daily_impressions = sum(hourly.values())
        
        summary = f"""
        🕐 **Hourly Performance Analysis**
        
        **Key Insights:**
        • **Peak Hour**: {peak_hour[0]} with {peak_hour[1]:,} impressions
        • **Lowest Hour**: {lowest_hour[0]} with {lowest_hour[1]:,} impressions
        • **Total Daily Impressions**: {total_daily_impressions:,}
        • **Average Hourly Impressions**: {total_daily_impressions/len(hourly):,.0f}
        
        **Peak Performance Windows:**
        • Morning Rush: 7:00 AM - 9:00 AM
        • Evening Rush: 6:00 PM - 8:00 PM
        • Late Night: Lower but consistent traffic
        """
        
        # Create hourly visualization
        hours = list(hourly.keys())
        impressions = list(hourly.values())
        
        fig = go.Figure()
        
        # Add line chart
        fig.add_trace(go.Scatter(
            x=hours, y=impressions,
            mode='lines+markers',
            name='Hourly Impressions',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8, color='#e74c3c')
        ))
        
        # Add peak hour annotation
        fig.add_annotation(
            x=peak_hour[0], y=peak_hour[1],
            text=f"Peak: {peak_hour[1]:,}",
            showarrow=True,
            arrowhead=2,
            bgcolor="yellow",
            bordercolor="red"
        )
        
        fig.update_layout(
            title="🕒 Dell Campaign - Hourly Impression Patterns",
            xaxis_title="Hour of Day",
            yaxis_title="Impressions",
            template="plotly_white",
            height=500
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return summary, fig
    
    def get_network_ids_info(self) -> str:
        """Get network IDs information"""
        network_ids = self.report_data["network_ids"]
        
        info = f"""
        🌐 **Network IDs for Dell Campaign**
        
        **Total Networks**: {len(network_ids)} screens
        
        **Network ID List:**
        """
        
        # Group IDs for better display
        for i in range(0, len(network_ids), 3):
            group = network_ids[i:i+3]
            info += f"\n• {' | '.join(group)}"
        
        info += f"""
        
        **ID Pattern Analysis:**
        • Prefix: JPN-JEK-D (Japan Digital Network)
        • Sequential numbering from 00029 to 00061
        • Some IDs skipped (strategic placement)
        """
        
        return info
    
    def get_glossary_info(self) -> str:
        """Get glossary and definitions"""
        glossary = self.report_data["glossary_notes"]
        
        info = f"""
        📚 **Campaign Glossary & Definitions**
        
        **Key Metrics:**
        """
        
        for term, definition in glossary.items():
            info += f"\n• **{term}**: {definition}"
        
        info += f"""
        
        **Additional Terms:**
        • **DOOH**: Digital Out-of-Home advertising
        • **JAD Vision**: Digital advertising network in Japan
        • **Network ID**: Unique identifier for each digital screen location
        • **Campaign Period**: {self.report_data['report_info']['Campaign_Duration']['Start']} to {self.report_data['report_info']['Campaign_Duration']['End']}
        """
        
        return info
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query and return appropriate response"""
        
        # Parse the query
        query_analysis = self.parse_query(user_query)
        
        if not query_analysis["is_jad_vision_query"]:
            return {
                "success": False,
                "message": query_analysis["message"],
                "suggestion": "Try asking: 'JAD Vision 7th July - 3rd Aug Dell report info' or 'Show JAD Vision Dell performance summary'"
            }
        
        query_type = query_analysis["query_type"]
        response = {"success": True, "query_type": query_type}
        
        # Route to appropriate handler
        if query_type == "report_info":
            response["details"] = self.get_report_info_details()
            response["title"] = "📊 Dell Campaign Report Information"
            
        elif query_type == "screens":
            response["details"] = self.get_screen_details_info()
            response["title"] = "🖥️ Screen Network Details"
            
        elif query_type == "performance":
            details, chart = self.get_performance_summary()
            response["details"] = details
            response["chart"] = chart
            response["title"] = "📈 Performance Summary"
            
        elif query_type == "demographics":
            details, charts = self.get_demographics_info()
            response["details"] = details
            response["charts"] = charts
            response["title"] = "👥 Demographic Analysis"
            
        elif query_type == "hourly":
            details, chart = self.get_hourly_analysis()
            response["details"] = details
            response["chart"] = chart
            response["title"] = "🕐 Hourly Performance Analysis"
            
        elif query_type == "network_ids":
            response["details"] = self.get_network_ids_info()
            response["title"] = "🌐 Network IDs Information"
            
        elif query_type == "glossary":
            response["details"] = self.get_glossary_info()
            response["title"] = "📚 Glossary & Definitions"
            
        else:  # general query
            # Provide overview of available information
            response["details"] = f"""
            📋 **JAD Vision Dell Campaign Overview**
            
            **Available Information Sections:**
            • **Report Info**: Campaign duration, technical details
            • **Screen Details**: Network specifications and coverage
            • **Performance Summary**: Impressions, reach, and frequency data
            • **Demographics**: Age and gender breakdown
            • **Hourly Analysis**: Time-based performance patterns
            • **Network IDs**: Complete list of screen identifiers
            • **Glossary**: Definitions and terminology
            
            **Quick Stats:**
            • Campaign Period: July 7 - August 3, 2024
            • Total Networks: 25 screens
            • Coverage: Major stations across Japan
            
            **Ask me specifically about any section above!**
            """
            response["title"] = "📊 JAD Vision Dell Campaign"
        
        return response

def create_jad_vision_interface():
    """Create the JAD Vision report interface"""
    
    st.set_page_config(
        page_title="JAD Vision Dell Campaign Report",
        page_icon="📊", 
        layout="wide"
    )
    
    # Custom styling
    st.markdown("""
    <style>
    .main-header {
        color: #2E86AB;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .query-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📊 JAD Vision Dell Campaign Report System</h1>', unsafe_allow_html=True)
    st.markdown("### Query system for Dell Digital Out-of-Home campaign data")
    
    # Initialize system
    if 'jad_system' not in st.session_state:
        st.session_state.jad_system = JADVisionReportSystem()
    
    # Sidebar with example queries
    st.sidebar.header("📝 Example Queries")
    example_queries = [
        "JAD Vision 7th July - 3rd Aug Dell report info",
        "JAD Vision Dell performance summary", 
        "Show JAD Vision Dell demographics",
        "JAD Vision Dell hourly analysis",
        "JAD Vision Dell network IDs",
        "JAD Vision Dell screen details",
        "JAD Vision Dell glossary definitions"
    ]
    
    selected_example = st.sidebar.selectbox("Choose an example:", [""] + example_queries)
    
    # Main query interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "🔍 Enter your query about JAD Vision Dell campaign:",
            value=selected_example if selected_example else "",
            placeholder="e.g., JAD Vision 7th July - 3rd Aug Dell performance summary"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("📊 Get Report Data", type="primary")
    
    # Process query when button is clicked or example is selected
    if (search_button and user_query) or (selected_example and selected_example != ""):
        query_to_process = user_query if user_query else selected_example
        
        with st.spinner("🔍 Searching campaign data..."):
            try:
                response = st.session_state.jad_system.process_query(query_to_process)
                
                if response["success"]:
                    # Display title
                    st.subheader(response["title"])
                    
                    # Display details
                    st.markdown(f'<div class="info-card">{response["details"]}</div>', unsafe_allow_html=True)
                    
                    # Display charts if available
                    if "chart" in response:
                        st.plotly_chart(response["chart"], use_container_width=True)
                    
                    if "charts" in response:
                        if len(response["charts"]) > 1:
                            # Display charts in columns
                            cols = st.columns(2)
                            for i, chart in enumerate(response["charts"]):
                                with cols[i % 2]:
                                    st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.plotly_chart(response["charts"][0], use_container_width=True)
                    
                else:
                    st.warning(response["message"])
                    if "suggestion" in response:
                        st.info(f"💡 {response['suggestion']}")
                        
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
    
    # Information panel
    st.markdown("---")
    
    # Quick stats in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Campaign Duration", "28 days")
    with col2:
        st.metric("🖥️ Total Screens", "25 networks")
    with col3:
        st.metric("🎯 Spot Length", "15 seconds")
    with col4:
        st.metric("🌏 Coverage", "Japan Major Stations")
    
    # Help section
    with st.expander("❓ How to Use This System"):
        st.markdown("""
        **Query Format:** Always include "JAD Vision" and "Dell" in your query for best results.
        
        **Available Information:**
        1. **Report Info**: Campaign dates, duration, technical details
        2. **Performance**: Impressions, reach, frequency by location
        3. **Demographics**: Age and gender breakdown
        4. **Hourly**: Time-based performance patterns
        5. **Screen Details**: Network specifications
        6. **Network IDs**: Complete list of screen identifiers
        7. **Glossary**: Definitions and terminology
        
        **Example Queries:**
        - "JAD Vision Dell report info" - Get campaign overview
        - "JAD Vision Dell performance" - See performance metrics
        - "JAD Vision Dell demographics" - View audience breakdown
        - "JAD Vision Dell hourly" - Analyze time patterns
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📊 JAD Vision Dell Campaign Report System | Digital Out-of-Home Analytics</p>
        <p>Campaign Period: July 7 - August 3, 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_jad_vision_interface()
