# To run: streamlit run src/streamlit_app.py

import streamlit as st
import requests

MODEL_API_URL = "http://127.0.0.1:8000/search"

st.title("Document Search")

query = st.text_input("Enter your search query:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        # Send request to FastAPI
        url = MODEL_API_URL
        payload = {"query": query}

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
