import streamlit as st
from summarizer import get_summary
from docHandling import get_text_from_file
from storage_and_retrieval import chunk_text, store_chunks, db, reload_connection
from queryProcessor import get_query_response
import os
import atexit
import warnings
import nest_asyncio
warnings.filterwarnings("ignore", message="CropBox missing from /Page")
nest_asyncio.apply()

st.set_page_config(page_title="RAG App", layout="centered")
st.title("Retrieval-Augmented Generation (RAG) App")

# --- Session State ---
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""
if "query_output" not in st.session_state:
    st.session_state.query_output = ""

# --- File Path Input Instead of Upload ---
st.subheader("📁 Provide File Path")
file_path = st.text_input("Paste the full path to your file (.txt, .pdf, .md):")
# remove the surrounding quotes if any
file_path = file_path.strip('"').strip("'")
if st.button("📂 Load File"):
    if not file_path:
        st.warning("⚠️ Please paste a valid file path.")
    elif not os.path.exists(file_path):
        st.error("❌ File does not exist at the provided path.")
    else:
        try:
                text = get_text_from_file(file_path)
                chunks, metadatas = chunk_text(text, os.path.basename(file_path))
                if db: 
                    reload_connection()
                store_chunks(chunks, metadatas)
                st.session_state.file_processed = True
                st.success(f"✅ File '{os.path.basename(file_path)}' loaded and processed successfully.")
        except Exception as e:
            st.error(f"⚠️ Failed to load file: {str(e)}")

# --- Summary Generation ---
st.subheader("📝 Generate Summary")
if st.button("Generate Summary"):
    if st.session_state.file_processed:
        st.session_state.summary_output = get_summary()
    else:
        st.warning("📄 Please load and process a file first.")

st.text_area("📌 Summary Output", st.session_state.summary_output, height=300)

# --- Query Section ---
st.divider()
st.subheader("💬 Ask a Query")
query = st.text_input("Type your question here:")

if st.button("🔎 Submit Query"):
    if not query.strip():
        st.warning("Please enter a query first!")
    else:
        st.session_state.query_output = get_query_response(query)

st.text_area("🧠 Query Response", st.session_state.query_output, height=300)

# --- Footer ---
st.markdown("---")

# --- Cleanup ---
def cleanup():
    db.delete_collection()
    print("🧹 Database collection deleted.")

atexit.register(cleanup)
