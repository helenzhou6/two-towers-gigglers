# To run: streamlit run src/streamlit_app.py

import streamlit as st
import requests
import os

API_URL = os.getenv('API_URL', "http://localhost:8000")
HEALTH_URL = f"{API_URL}/health-check"

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #add8e6, #ffb6c1, #add8e6, #ffb6c1);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    </style>
""", unsafe_allow_html=True)

def check_api_status():
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

api_available = check_api_status()

if api_available:
    st.success(f"✅ API is running at {API_URL}")
else:
    st.error(f"❌ API is not available at {API_URL}. Please check the server and refresh.")

st.title("🔍 Doci-bot 🤖")

query = st.text_input("Enter your search query:")

# Disable Search button if API is down
if st.button("Search", disabled=not api_available):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        url = f"{API_URL}/search"
        payload = {"query": query}

        with st.spinner("Searching for documents..."):
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                results = response.json()

                # Display results
                if results:
                    st.success("Top matching documents found:")
                    for item in results:
                        st.markdown(f"**Rank {item['rank']}**")
                        st.write(item["doc"])
                        st.write("Similarity Score:", item["score"])
                        st.markdown("---")
                else:
                    st.info("No matching documents found.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error contacting API: {e}")
