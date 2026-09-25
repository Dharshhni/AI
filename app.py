import os
import streamlit as st
import pandas as pd

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart E-Learning RAG",
    page_icon="🎓",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #cccccc;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# TITLE
# =====================================================

st.markdown(
    '<div class="title">🎓 Smart E-Learning RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload study material, ask questions and generate quizzes'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Settings")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password"
)

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key


# =====================================================
# SESSION STATE
# =====================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []


# =====================================================
# FILE UPLOAD
# =====================================================

st.header("📚 Upload Study Material")

uploaded_file = st.file_uploader(
    "Upload PDF, TXT, CSV or Excel file",
    type=["pdf", "txt", "csv", "xlsx"]
)


# =====================================================
# FILE READING FUNCTIONS
# =====================================================

def read_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def read_txt(file):

    return file.read().decode("utf-8")


def read_csv(file):

    df = pd.read_csv(file)

    return df.to_string(index=False)


def read_excel(file):

    df = pd.read_excel(file)

    return df.to_string(index=False)


# =====================================================
# PROCESS FILE
# =====================================================

if uploaded_file:

    try:

        file_name = uploaded_file.name.lower()

        if file_name.endswith(".pdf"):
            text = read_pdf(uploaded_file)

        elif file_name.endswith(".txt"):
            text = read_txt(uploaded_file)

        elif file_name.endswith(".csv"):
            text = read_csv(uploaded_file)

        elif file_name.endswith(".xlsx"):
            text = read_excel(uploaded_file)

        else:
            text = ""


        if not text.strip():

            st.error(
                "❌ Could not extract information from the file."
            )

            st.stop()


        st.success(
            f"✅ {uploaded_file.name} loaded successfully!"
        )


        # =================================================
        # TEXT CHUNKING
        # =================================================

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        st.info(
            f"📄 Created {len(chunks)} text chunks."
        )


        # =================================================
        # EMBEDDINGS
        # =================================================

        with st.spinner(
            "🧠 Creating knowledge database..."
        ):

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vector_db = FAISS.from_texts(
                chunks,
                embeddings
            )

            st.session_state.vector_db = vector_db
            st.session_state.chunks = chunks


        st.success(
            "✅ Study material is ready!"
        )


    except Exception as e:

        st.error(
            "❌ Error while processing file."
        )

        st.code(str(e))


# =====================================================
# RAG FEATURES
# =====================================================

if st.session_state.vector_db is not None:

    st.divider()

    tab1, tab2 = st.tabs(
        ["💬 Ask Questions", "📝 Generate Quiz"]
    )


    # =================================================
    # QUESTION ANSWERING
    # =================================================

    with tab1:

        st.header("💬 Ask Your Study Material")

        question = st.text_input(
            "Enter your question:",
            placeholder="Example: What is Machine Learning?"
        )


        if st.button(
            "🔍 Ask Question",
            use_container_width=True
        ):

            if not api_key:

                st.warning(
                    "⚠️ Please enter your Gemini API key."
                )

            elif not question:

                st.warning(
                    "⚠️ Please enter a question."
                )

            else:

                with st.spinner(
                    "🔎 Searching study material..."
                ):

                    documents = (
                        st.session_state.vector_db
                        .similarity_search(
                            question,
                            k=4
                        )
                    )

                    context = "\n\n".join(
                        doc.page_content
                        for doc in documents
                    )


                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.8-flash",
                        temperature=0.2
                    )


                    prompt = f"""
You are an E-Learning AI assistant.

Answer the student's question using ONLY
the information provided in the study material.

Explain the answer in a simple,
student-friendly way.

Do not invent information.

If the answer is not available, say:

"I could not find this information in the
uploaded study material."

Study Material:
--------------------
{context}
--------------------

Student Question:
{question}

Give a clear and simple answer.
"""


                    response = llm.invoke(prompt)

                    answer = response.content


                st.session_state.chat_history.append(
                    (question, answer)
                )


                st.subheader("🤖 AI Answer")

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.subheader(
                    "📚 Retrieved Study Material"
                )

                for i, doc in enumerate(documents):

                    with st.expander(
                        f"Source {i + 1}"
                    ):

                        st.write(
                            doc.page_content
                        )


        # =================================================
        # CHAT HISTORY
        # =================================================

        if st.session_state.chat_history:

            st.divider()

            st.subheader("🕘 Previous Questions")

            for q, a in reversed(
                st.session_state.chat_history
            ):

                with st.expander(q):

                    st.write(a)


        if st.button("🗑️ Clear Chat"):

            st.session_state.chat_history = []

            st.rerun()


    # =================================================
    # QUIZ GENERATOR
    # =================================================

    with tab2:

        st.header("📝 Generate Quiz")

        number_of_questions = st.slider(
            "Number of Questions",
            3,
            10,
            5
        )


        if st.button(
            "🎯 Generate Quiz",
            use_container_width=True
        ):

            if not api_key:

                st.warning(
                    "⚠️ Please enter your Gemini API key."
                )

            else:

                with st.spinner(
                    "🧠 Creating quiz..."
                ):

                    quiz_context = "\n\n".join(
                        st.session_state.chunks[:10]
                    )


                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.8-flash",
                        temperature=0.3
                    )


                    quiz_prompt = f"""
Create {number_of_questions} multiple-choice
questions using ONLY the study material.

For every question provide:

Question:
A)
B)
C)
D)
Correct Answer:

Make the questions simple and suitable
for students.

Study Material:
--------------------
{quiz_context}
--------------------
"""


                    response = llm.invoke(
                        quiz_prompt
                    )

                    quiz = response.content


                st.subheader("🎯 Generated Quiz")

                st.markdown(quiz)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "🎓 Smart E-Learning RAG Assistant | "
    "Retrieval-Augmented Generation"
)

