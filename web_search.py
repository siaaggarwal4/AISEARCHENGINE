import streamlit as st
import json
import math
import re
import time
from nltk.stem import PorterStemmer
from searching import search_cli
stemmer = PorterStemmer()
 

st.title("Search Engine")
# st.write(f"{total_docs} documents indexed")

query = st.text_input("Enter query:")

if query:
    results, elapsed, nearsim = search_cli(query)
    st.write(f"{len(results)} results ({elapsed} ms)")
    if results:
        for i, (url, score) in enumerate(results, 1):
            st.write(f"{i}. {url}")
            st.caption(f"score: {score}")
    else:
        st.write("no results found")