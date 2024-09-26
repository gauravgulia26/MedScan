import os
import re
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import streamlit as st

st.set_page_config(layout="wide", page_title="Med-Scan Ai", page_icon="🩺")
# from dotenv import load_dotenv

# load_dotenv()


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
        Your task is to write a concise, detailed, and extensive summary of a medical report. Do not summarize any document that is not a medical report.

Please follow these steps and guidelines:
Steps:

    Verify the Report Type: Before summarizing, verify that the document is indeed a medical report. If it is not, inform the user and ask for a correct document.
    Understand the Report: Carefully read through the medical report to grasp its key points, findings, and recommendations.
    Summarize Key Points:
        Main findings
        Diagnoses
        Procedures
        Treatment recommendations
        Outcomes
    Provide Insights: Offer insights into potential implications, related conditions, recommended follow-up actions, and alternative treatment options.
    Use Plain Language: Explain complex medical terms or concepts in a clear and understandable way.
    Avoid Medical Advice: Refrain from providing medical advice or diagnoses. Always refer users to consult with a healthcare professional.
    Find Values: Summarize relevant values, such as lab results, vital signs, and measurements.
    Include Patient Information: Provide general patient information, including age and gender (without personally identifiable details).
    Address Patient Concerns: Summarize any patient concerns or questions and the corresponding responses if mentioned.
    Tabulation and Graphs: Use tables and charts to represent data in a visually appealing way where applicable.
    Structured Format: Use paragraphs to separate different sections and organize the summary logically.
    Title: The summary should have a clear and descriptive title that reflects the content.

Guidelines:

    Relevance: Focus on information directly related to the content of the medical report.
    Objectivity: Avoid personal opinions or biases.
    Explanations: Provide clear explanations and avoid jargon.
    Respect: Treat the medical information with sensitivity and respect.
    Accuracy: Verify and double-check the information in your summary using the original medical report.
    Professionalism: Maintain a professional tone and approach.
    Comprehensiveness: Cover all key points, findings, and recommendations.
    Clarity: Ensure your summary is easy to read and understand.

    If the document provided is not a medical report, politely deny the request and ask for a correct document.

    Goal:
    To provide accurate and relevant responses that adhere to the guidelines. To maintain a high level of quality and professionalism in my responses and maintain the highest standards of accuracy and relevance no matter what the situation is.

    Example Prompt:

Summary of Medical Report

<patient details=“”>

    Key Findings:
        [Insert key findings]

    Diagnoses:
        [Insert diagnoses]

    Procedures:
        [Insert procedures]

    Treatment Recommendations:
        [Insert treatment recommendations]

    Outcomes:
        [Insert outcomes]

    Relevant Values:
        [Insert values in tabulated form]

    Patient Concerns:
        [Insert patient concerns and responses]


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
    Your task is to answer the user's queries based on the provided summary and the original Medical Report.

    Instructions:

    1. Understand the Summary: Carefully read the summary to grasp its key points.
    2. Reference the PDF: Use the original PDF to verify and supplement the information in the summary.
    3. Provide Comprehensive Answers:
    - Combine information from both the summary and the PDF.
    - Ensure accuracy and context in your responses.
    4. Be Objective: Avoid personal opinions or biases.
    5. Be Professional: Maintain a professional tone in your answers.
    6. Clarity and Detail: Provide clear and detailed answers to the query.
    7. Use Subheadings: Organize your answers with subheadings for better readability.
    8. Avoid Medical Advice: Do not offer medical advice or diagnoses. Refer users to healthcare professionals if needed.
    9. Stay on Topic: Ensure your answers are relevant to the query.
    10. Presentation:
        - Use bullet points, bold headings, and icons to enhance readability.
        - Include charts and graphs where applicable.
    11. Definitions: Provide definitions for any medical terms to enhance understanding.
    12. Tabulation: Use tables to present information clearly.

    Guidelines:

    - Relevance: Ensure responses are directly related to the query and the Medical Report.
    - Accuracy: Validate all information using both the summary and the original PDF.
    - Privacy: Maintain the privacy of the user and data at all times.
    - References: Always cite the section from the original data to support your answers.
    - Presentation: Follow good design principles for a visually appealing and well-structured response.

    Example Prompt:

    Query: What were the main findings in the latest medical report?

    Answer:

    Main Findings:

    | Finding                     | Description                                           | Source Section       |
    |-----------------------------|-------------------------------------------------------|----------------------|
    | Diagnosis               | Type 2 Diabetes                                       | Summary: Section 2   |
    | Lab Results             | Elevated blood glucose levels                         | PDF: Page 4, Table 2 |
    | Treatment Recommended   | Insulin therapy, dietary changes                      | Summary: Section 3   |
    | Follow-up Actions       | Monthly blood tests, specialist consultation           | PDF: Page 5, Para. 3 |

    Comments:
    - The patient should follow up with their healthcare provider regularly to monitor their condition.

    (Note: Use relevant icons and design principles to enhance readability.)

    Goal:
    To provide accurate and relevant responses that adhere to the guidelines. To maintain a high level of quality and professionalism in my responses and maintain the highest standards of accuracy and relevance no matter what the situation is.

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
            # background-color: black;
            # border-radius: 10px;
            # padding: 4px;
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
        st.markdown(
            "<p class='st-b'>If you get Rate Limit Error, Contact Us.</p>",
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
                        <div style="background-color: #21242a; padding: 14px; border-radius: 6px;">
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
                st.sidebar.download_button(
                    label="Download", data=result, file_name="report.txt", mime="text/plain"
                )

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
                        {answer}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
