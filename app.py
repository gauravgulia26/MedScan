import os
import re
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfReader
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import streamlit as st

st.set_page_config(layout="wide", page_title="MedScan", page_icon="🩺")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from streamlit_extras.streaming_write import write
import time
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# Disable caching on data ingestion and text cleaning functions
def data_ingestion(url):
    pdf_reader = PdfReader(url)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def clean_text(text):
    """Cleans a single text by removing common escape sequences, unnecessary characters, and empty elements."""
    cleaned_text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    cleaned_text = re.sub(r"\u200b", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def cleaning_pipeline(_data):
    """Cleans the text in the document."""
    return clean_text(_data)


# Cache embeddings and models for performance
@st.cache_resource(show_spinner=False)
def embeddings_and_model():
    extrct_embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HUGGINGFACE_APIKEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    extrct_model = ChatGroq(
        model="llama-3.1-8b-instant",
        max_tokens=7000,
        temperature=0.3,
        streaming=True,
        model_kwargs={"top_p": 0.5},
        api_key=os.getenv("GROQ_API_KEY_EXTRCT"),
    )
    summ_model = ChatGroq(
        model="llama-3.1-70b-versatile",
        max_tokens=7000,
        temperature=0.3,
        streaming=True,
        model_kwargs={"top_p": 0.5},
        api_key=os.getenv("GROQ_API_KEY_SUMM"),
    )
    return extrct_embeddings, extrct_model, summ_model


extrct_embeddings, extrct_model, summ_model = embeddings_and_model()


def vector_db(_corpus):
    extrct_db = FAISS.from_texts(_corpus, extrct_embeddings)
    extrct_db = extrct_db.as_retriever()
    return extrct_db


def prompt_chain(_data):
    summ_prompt = ChatPromptTemplate.from_template(
        """
        Task: Write a concise, detailed, and extensive summary of the following medical report.

    Instructions:

    1. **Understand the Report:** Carefully read the medical report to grasp its key points, findings, and recommendations.
    2. **Summarize Key Points:** Concisely summarize the main findings, diagnoses, procedures, treatment recommendations, and outcomes.
    3. **Provide Insights:** Offer insights into potential implications, related conditions, recommended follow-up actions, and alternative treatment options.
    4. **Generate Questions:** Ask relevant questions that could help clarify the report's content, suggest further investigation, or address potential concerns.
    5. **Use Plain Language:** Explain complex medical terms or concepts in a clear and understandable way.
    6. **Avoid Medical Advice:** Refrain from providing medical advice or diagnoses. Always refer users to consult with a healthcare professional.
    7. **Find Values:** Summarize relevant values, such as lab results, vital signs, and measurements.
    8. **Include Patient Information:** Provide general patient information, including age and gender (without personally identifiable details).
    9. **Address Patient Concerns:** Summarize any patient concerns or questions and the corresponding responses if mentioned.
    10. Tabulation: Use tabulation to organize the summary in a structured manner.
    11. Bullet Points: Use bullet points to list key findings, diagnoses, and recommendations.
    12. Title: The summary should have a clear and descriptive title that reflects the content.
    13. Patient Information: Extract all the relevant patient information from the report including the doctor and hospital details.

    Guidelines:

    - **Focus on Relevance:** Ensure your responses are directly related to the content of the medical report.
    - **Be Objective:** Avoid personal opinions or biases in your analysis.
    - **Provide Clear Explanations:** Use simple language and avoid jargon.
    - **Be Respectful:** Treat the medical information with sensitivity and respect.

    The medical report is enclosed below:
    {context}
    """
    )
    summ_chain = summ_prompt | summ_model | StrOutputParser()
    result = summ_chain.invoke({"context": _data})
    result = re.sub(r"\*", "", result)
    return result


@st.cache_resource(show_spinner=False)
def extract_chain(_model, _extrct_prompt, _extrct_db):
    _extrct_prompt = ChatPromptTemplate.from_template(
        """
    Task: Answer the user's queries based on the provided summary and the original Medical Report.

    Instructions:

    1. **Understand the Summary:** Carefully read the summary to grasp its key points.
    2. **Reference the PDF:** Use the original PDF to verify and supplement the information in the summary.
    3. **Provide Comprehensive Answers:** Combine the information from both the summary and the PDF to provide accurate and informative responses.
    4. **Be Objective:** Avoid personal opinions or biases in your responses.
    5. **Be Accurate:** Verify the information in your responses using both sources.
    6. **Be Professional:** Maintain a professional tone and approach in your responses.
    7. **Be Respectful:** Treat the medical information with sensitivity and respect.
    8. **Provide Context:** Your answers should contain the section of the report the information is derived from.
    9. **Clarity and Detail:** Provide clear and detailed information to answer the query comprehensively.
    10. **Use Subheadings:** Organize your answers with subheadings to enhance readability.
    11. **Avoid Medical Advice:** Refrain from providing medical advice or diagnoses. Always refer users to consult with a healthcare professional.
    12. **Stay on Topic:** Always stay relevant to the query and avoid out-of-context or out-of-scope answers.
    13. **Greeting Style:** Start with a greeting, then provide the answer in a detailed manner.
    14. Comments: Your answer should also contain final comments about the query.
    15. UI: Use good ui design principles like bullet points, bold headings and many more to make the summary visually appealing and easy to read.
    16. Icons: Provide relating icons for different sections to make the summary more engaging and visually appealing.
    17. Charts and Graphs: Use charts and graphs to represent data in a more visually appealing way.
    18. References: Always provide references to the original data to support your answers
    19. Definitions: Provide definitions for medical terms and jargon to make the summary more understandable if applicable.
    20. Designing: Use good designing principles to make the summary visually appealing and easy to read.
    21.Tabulation: Your response should contains output in tabulation format where necessary.
    22. Focus on presentation: Your response should be well-structured and visually appealing to the reader.

    Guidelines:

    - **Focus on Relevance:** Ensure your responses are directly related to the query and the summary/Medical Report.
    - **Be Clear and Detailed:** Explain your answers in an understandable manner with sufficient detail.
    - **Provide References:** Always reference the original data to support your answers.
    - **Respect Privacy:** Maintain the privacy of the user and the data at all times.

    Your context is as follows:
    <context>
    Summary:
    {summary}

    Original PDF:
    {context}
    </context>

    Question: {input}
    """
    )

    extrct_chain = create_stuff_documents_chain(_model, _extrct_prompt)
    retrieval_chain = create_retrieval_chain(_extrct_db, extrct_chain)
    return retrieval_chain


# Main function
def main():
    # Custom CSS to hide the GitHub icon
    hide_github_icon = """
        <style>
        #MainMenu {visibility: hidden;} /* Hides the hamburger menu */
        footer {visibility: hidden;} /* Hides the footer with GitHub icon */
        .viewerBadge_container__1QSob {visibility: hidden;} /* Hides the Streamlit viewer badge */
        </style>
        """

    # Injecting custom CSS
    st.markdown(hide_github_icon, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            """
        <style>
        .st-b {
            font-size: 12px;
            color: lime;
            background-color: black;
            border-radius: 10px;
            padding: 4px;
            text-align: center;
        }
        .st-r {
            font-size: 11px;
            color: lime;
            background-color: black;
            border-radius: 14px;
            padding: 3px;
            text-align: center;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='st-r'>Experimental Release. May contains some limitations.</p>",
            unsafe_allow_html=True,
        )
        st.sidebar.title("👩‍⚕️ MedScan")
        st.sidebar.markdown(
            """
            Your Digital Health Assistant.

            Upload your medical document and get a detailed summary ready to be downloaded.

            Ask follow-up questions to get more insights.
            """,
            unsafe_allow_html=True,
        )

        st.link_button("Contact Us", "mailto:grvgulia007@gmail.com")

    st.title("MedScan, Your Personal Health Assistant.")
    # Clear cached summary when a new PDF is uploaded
    if "prev_file" not in st.session_state:
        st.session_state["prev_file"] = None
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf is not None:
        # If the uploaded file is different from the previous one, reset cache
        if st.session_state["prev_file"] != pdf:
            st.session_state["prev_file"] = pdf
            st.session_state["corpus"] = None
            st.session_state["summary"] = None

        if "corpus" not in st.session_state or st.session_state["corpus"] is None:
            # Process the PDF document
            corpus = data_ingestion(pdf)
            corpus = cleaning_pipeline(corpus)
            corpus = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_text(
                corpus
            )
            st.session_state["corpus"] = corpus
            st.session_state["extrct_db"] = vector_db(corpus)

        # Generate summary if not already generated
        if "summary" not in st.session_state or st.session_state["summary"] is None:
            with st.spinner("Analysing and Generating Report"):
                result = prompt_chain(st.session_state["corpus"])
                st.session_state["summary"] = result
                with st.chat_message("🩺"):
                    st.markdown(
                        f"""
                        <div style="background-color: #21242a; padding: 10px; border-radius: 5px;">
                        <h4>Summary</h4>
                        {result}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write("\n")
                    st.markdown(
                        "<p class='st-b'>  Your Summary Report has been Created, You can download it using the sidebar.</p>",
                        unsafe_allow_html=True,
                    )
                st.sidebar.download_button(label="Download", data=result, file_name="report.txt")

        # Answer follow-up questions
        retrieval_chain = extract_chain(
            extrct_model, st.session_state["summary"], st.session_state["extrct_db"]
        )
        question = st.chat_input("Ask follow-up questions here...")

        if question:
            with st.spinner("Thinking..."):
                answer = retrieval_chain.invoke(
                    {"input": question, "summary": st.session_state["summary"]}
                )["answer"]
                answer = re.sub(r"\*", "", answer)
                with st.chat_message("🩺"):
                    st.markdown(
                        f"""
                        <div style="background-color: #21242a; padding: 10px; border-radius: 5px;">
                        <h4>Answer</h4>
                        {answer}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
