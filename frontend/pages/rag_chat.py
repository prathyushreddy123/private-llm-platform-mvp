"""
RAG Chat Interface for Legal Document Q&A

This Streamlit page provides an interface for querying legal documents
using the RAG (Retrieval-Augmented Generation) pipeline.
"""

import os
import json
import time
import httpx
import streamlit as st
from typing import List, Dict, Any, Optional

# Configure page
st.set_page_config(
    page_title="Legal RAG Chat",
    page_icon="⚖️",
    layout="wide"
)

# Constants
API_URL = os.environ.get("RAG_API_URL", "http://localhost:8080")
DEFAULT_COLLECTION = "huggingface_collection"
DOMAINS = ["legal", "default", "technical", "medical"]

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_collection" not in st.session_state:
    st.session_state.current_collection = DEFAULT_COLLECTION
if "collections" not in st.session_state:
    st.session_state.collections = []
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "show_context" not in st.session_state:
    st.session_state.show_context = False
if "domain" not in st.session_state:
    st.session_state.domain = "legal"

# Helper functions
@st.cache_data(ttl=300)
def get_collections() -> List[str]:
    """Get available collections from the API."""
    try:
        response = httpx.get(f"{API_URL}/collections", timeout=10.0)
        if response.status_code == 200:
            return response.json()["collections"]
        else:
            st.error(f"Failed to get collections: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

def query_rag(query: str, collection: str, domain: str) -> Dict[str, Any]:
    """Query the RAG API."""
    try:
        # Prepare the request payload
        payload = {
            "query": query,
            "collection": collection,
            "domain": domain,
            "chat_history": st.session_state.chat_history[-5:] if len(st.session_state.chat_history) > 0 else []
        }
        
        # Add retrieval options if set
        if "retrieval_options" in st.session_state:
            payload["retrieval_options"] = st.session_state.retrieval_options
        
        # Add generation options if set
        if "generation_options" in st.session_state:
            payload["generation_options"] = st.session_state.generation_options
        
        # Make the request
        with st.spinner("Thinking..."):
            start_time = time.time()
            response = httpx.post(
                f"{API_URL}/query",
                json=payload,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                result["elapsed_time"] = time.time() - start_time
                return result
            else:
                st.error(f"Error from API: {response.text}")
                return {
                    "response": f"Error: {response.text}",
                    "confidence_score": 0.0,
                    "elapsed_time": time.time() - start_time
                }
    except Exception as e:
        st.error(f"Error querying RAG API: {str(e)}")
        return {
            "response": f"Error: {str(e)}",
            "confidence_score": 0.0,
            "elapsed_time": 0.0
        }

def format_chat_message(is_user: bool, message: str, metadata: Optional[Dict[str, Any]] = None):
    """Format a chat message with proper styling."""
    if is_user:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background-color: #2b313e; color: white; border-radius: 15px; padding: 10px 15px; max-width: 80%;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence = metadata.get("confidence_score", 0) if metadata else 0
        elapsed_time = metadata.get("elapsed_time", 0) if metadata else 0
        
        # Color based on confidence
        confidence_color = "#4CAF50" if confidence > 0.8 else "#FFC107" if confidence > 0.5 else "#F44336"
        
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
            <div style="background-color: #0e1117; color: white; border-radius: 15px; padding: 10px 15px; max-width: 80%;">
                {message}
                <div style="margin-top: 5px; font-size: 0.8em; color: #888;">
                    <span style="color: {confidence_color}">Confidence: {confidence:.2f}</span> | Response time: {elapsed_time:.2f}s
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_context(context, metadata):
    """Display the retrieved context with source information."""
    st.markdown("### Retrieved Context")
    
    for i, (text, meta) in enumerate(zip(context, metadata)):
        with st.expander(f"Source {i+1}", expanded=i==0):
            st.markdown(f"**Text:** {text}")
            st.markdown("**Metadata:**")
            st.json(meta)

def display_citations(citations):
    """Display citations from the response."""
    if not citations:
        st.info("No citations found in the response.")
        return
    
    st.markdown("### Citations")
    for i, citation in enumerate(citations):
        with st.expander(f"Citation {i+1}", expanded=i==0):
            st.markdown(f"**Text:** {citation['text']}")
            st.markdown("**Source:**")
            st.json(citation['source_metadata'])

# UI Components
def render_sidebar():
    """Render the sidebar with settings."""
    st.sidebar.title("⚖️ Legal RAG Chat")
    
    # Collection selector
    st.sidebar.subheader("Collection")
    collections = get_collections()
    st.session_state.collections = collections
    
    if collections:
        selected_collection = st.sidebar.selectbox(
            "Select Collection",
            options=collections,
            index=collections.index(st.session_state.current_collection) if st.session_state.current_collection in collections else 0
        )
        st.session_state.current_collection = selected_collection
    else:
        st.sidebar.error("No collections available")
    
    # Domain selector
    st.sidebar.subheader("Domain")
    selected_domain = st.sidebar.selectbox(
        "Select Domain",
        options=DOMAINS,
        index=DOMAINS.index(st.session_state.domain) if st.session_state.domain in DOMAINS else 0
    )
    st.session_state.domain = selected_domain
    
    # Advanced settings
    st.sidebar.subheader("Advanced Settings")
    show_settings = st.sidebar.checkbox("Show Advanced Settings", value=st.session_state.show_settings)
    st.session_state.show_settings = show_settings
    
    if show_settings:
        # Retrieval settings
        st.sidebar.markdown("#### Retrieval Settings")
        
        if "retrieval_options" not in st.session_state:
            st.session_state.retrieval_options = {
                "top_k": 5,
                "threshold": 0.7,
                "rerank_threshold": 0.6,
                "max_tokens_limit": 3000
            }
        
        retrieval_options = st.session_state.retrieval_options
        
        top_k = st.sidebar.slider("Top K Results", 1, 20, retrieval_options["top_k"])
        threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, retrieval_options["threshold"])
        rerank_threshold = st.sidebar.slider("Reranking Threshold", 0.0, 1.0, retrieval_options["rerank_threshold"])
        max_tokens = st.sidebar.slider("Max Context Tokens", 1000, 10000, retrieval_options["max_tokens_limit"])
        
        st.session_state.retrieval_options = {
            "top_k": top_k,
            "threshold": threshold,
            "rerank_threshold": rerank_threshold,
            "max_tokens_limit": max_tokens
        }
        
        # Generation settings
        st.sidebar.markdown("#### Generation Settings")
        
        if "generation_options" not in st.session_state:
            st.session_state.generation_options = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500,
                "model": "llama2"
            }
        
        generation_options = st.session_state.generation_options
        
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, generation_options["temperature"])
        top_p = st.sidebar.slider("Top P", 0.0, 1.0, generation_options["top_p"])
        max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, generation_options["max_tokens"])
        model = st.sidebar.text_input("Model", value=generation_options["model"])
        
        st.session_state.generation_options = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "model": model
        }
    
    # Context display toggle
    show_context = st.sidebar.checkbox("Show Context & Citations", value=st.session_state.show_context)
    st.session_state.show_context = show_context
    
    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This chat interface uses RAG (Retrieval-Augmented Generation) "
        "to provide accurate answers based on legal documents."
    )

def render_chat():
    """Render the main chat interface."""
    st.title("Legal Document Q&A")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            format_chat_message(
                is_user=message["is_user"],
                message=message["content"],
                metadata=message.get("metadata")
            )
    
    # Display context and citations if enabled
    if st.session_state.show_context and st.session_state.chat_history:
        last_message = [m for m in st.session_state.chat_history if not m["is_user"]]
        if last_message and "metadata" in last_message[-1]:
            metadata = last_message[-1]["metadata"]
            if "context" in metadata and "context_metadata" in metadata:
                display_context(metadata["context"], metadata["context_metadata"])
            
            if "citations" in metadata:
                display_citations(metadata["citations"])
    
    # Input form
    with st.form(key="query_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question about legal documents:", height=100)
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "is_user": True,
                "content": user_input,
                "query": user_input
            })
            
            # Get response from RAG API
            response = query_rag(
                query=user_input,
                collection=st.session_state.current_collection,
                domain=st.session_state.domain
            )
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "is_user": False,
                "content": response["response"],
                "query": user_input,
                "response": response["response"],
                "metadata": {
                    "confidence_score": response.get("confidence_score", 0),
                    "elapsed_time": response.get("elapsed_time", 0),
                    "context": response.get("context", []),
                    "context_metadata": response.get("context_metadata", []),
                    "citations": response.get("citations", [])
                }
            })
            
            # Rerun to update the UI
            st.rerun()

# Main app
def main():
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()
