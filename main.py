import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time

# Page setup
st.set_page_config(page_title="Web Content Analyzer", layout="wide")

# Custom CSS for better formatting
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }

    .content-box {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }

    .bullet-point {
        background-color: #2C2C2C;
        border-left: 4px solid #FF4B4B;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Web Content Analyzer with Groq AI")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    st.info("Using Mixtral-8x7b model")

# URL input
url = st.text_input("Enter URL to analyze:", placeholder="https://example.com")


def setup_groq():
    """Initialize Groq LLM"""
    if not groq_api_key:
        st.error("Please enter your Groq API key!")
        st.stop()

    return ChatGroq(
        api_key=groq_api_key,
        model="mixtral-8x7b-32768",
        temperature=0.3
    )


def scrape_content(url):
    """Scrape content from URL using requests and BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text(separator='\n', strip=True)

        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    except Exception as e:
        st.error(f"Error scraping content: {str(e)}")
        return None


def format_bullet_points(text):
    """Convert text to properly formatted bullet points"""
    lines = text.strip().split('\n')
    formatted_html = "<div class='content-box'>"

    for line in lines:
        line = line.strip()
        if line:
            # Remove existing bullet points or dashes
            line = line.lstrip('•').lstrip('-').lstrip('*').strip()
            formatted_html += f"<div class='bullet-point'>{line}</div>"

    formatted_html += "</div>"
    return formatted_html


def analyze_url(url):
    """Main function to analyze URL content"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Scrape content
        status_text.text("Scraping webpage...")
        content = scrape_content(url)
        if not content:
            return None
        progress_bar.progress(0.25)

        # Step 2: Split text
        status_text.text("Processing content...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.create_documents([content])
        progress_bar.progress(0.50)

        # Step 3: Analyze with Groq
        status_text.text("Analyzing with Groq AI...")
        llm = setup_groq()

        summary_prompt = PromptTemplate(
            template="""Extract and summarize the key points from this text as a bullet-point list.
            Each point should start with a bullet point (*).
            Focus on meaningful information and ignore navigation elements.

            Text: {text}

            Key Points:""",
            input_variables=["text"]
        )

        chain = LLMChain(llm=llm, prompt=summary_prompt)

        all_summaries = []
        total_splits = len(splits)

        for i, split in enumerate(splits):
            summary = chain.invoke({"text": split.page_content})
            all_summaries.append(summary['text'])
            current_progress = 0.5 + (0.5 * (i + 1) / total_splits)
            progress_bar.progress(min(current_progress, 1.0))
            time.sleep(0.5)

        status_text.empty()
        progress_bar.empty()

        return all_summaries

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# Main execution
if st.button("Analyze", disabled=not url):
    if url:
        st.write("Starting analysis...")
        summaries = analyze_url(url)

        if summaries:
            # Display results in tabs
            tab1, tab2 = st.tabs(["Formatted Summary", "Raw Text"])

            with tab1:
                st.subheader("Analysis Results")
                for i, summary in enumerate(summaries, 1):
                    st.markdown(f"#### Section {i}")
                    # Use the new formatting function
                    formatted_html = format_bullet_points(summary)
                    st.markdown(formatted_html, unsafe_allow_html=True)

                # Add download button
                combined_text = "\n\n".join(summaries)
                st.download_button(
                    "Download Summary",
                    combined_text,
                    file_name="web_content_summary.txt",
                    mime="text/plain"
                )

            with tab2:
                st.text_area("Raw Summary Text", "\n\n".join(summaries), height=400)

        st.success("Analysis complete!")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Groq AI")