import streamlit as st
import json

def print_tab(title, data):
    st.header(title)
    if isinstance(data, list):
        for i, entry in enumerate(data, start=1):
            with st.expander(f"Entry {i}"):
                if isinstance(entry, dict):
                    for key, value in entry.items():
                        st.write(f"**{key}**: {value}")
                else:
                    st.write(entry)
    elif isinstance(data, dict):
        for key, value in data.items():
            st.write(f"**{key}**: {value}")
    else:
        st.write(data)

st.title("JSON Viewer with File Uploader")

uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

if uploaded_file is not None:
    try:
        json_data = json.load(uploaded_file)
        st.success("JSON loaded successfully!")

        # Print all tabs
        for tab_name, tab_data in json_data.items():
            print_tab(tab_name, tab_data)

    except Exception as e:
        st.error(f"Error loading JSON: {e}")
else:
    st.info("Please upload a JSON file to display contents.")
