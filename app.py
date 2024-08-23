import streamlit as st
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
import os
import random
import textwrap as tr
from text_load_utils import parse_txt, text_to_docs, parse_pdf
from db_chat import user_message, bot_message

st.set_page_config("Multilingual Chat Bot 🤖", layout="centered")
st.sidebar.text("Resources")
st.sidebar.markdown(
    """ 
- [Multilingual Embedding Models](https://docs.cohere.com/docs/multilingual-language-models)
- [Multilingual Search with Cohere and Langchain](https://github.com/cohere-ai/notebooks/blob/main/notebooks/Multilingual_Search_with_Cohere_and_Langchain.ipynb)
- [LangChain](https://python.langchain.com/en/latest/index.html)
"""
)
with st.sidebar:
    uploaded_file = st.file_uploader(
        "**Upload a pdf or txt file :**",
        type=["pdf", "txt"],
    )

# Cohere API Initiation
# cohere_api_key = st.secrets["COHERE_API_KEY"]
cohere_api_key = 'EjlUovaWWRpPT4PAPlOIRXWpfWByYByUqWJIPQxa'

st.title("Multilingual Chat Bot 🤖")


if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        doc = parse_txt(uploaded_file)
    else:
        doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)

    page_holder = st.empty()

    # Display the content of the uploaded file
    with page_holder.expander("File Content", expanded=False):
        pages

    # Create our own prompt template
    prompt_template = """Text: {context}

    Question: {question}

    Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available."""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    # Bot UI dump
    # Session State Initiation
    prompt = st.session_state.get("prompt", None)

    if prompt is None:
        prompt = [{"role": "system", "content": prompt_template}]

    for message in prompt:
        if message["role"] == "user":
            user_message(message["content"])
        elif message["role"] == "assistant":
            bot_message(message["content"], bot_name="Multilingual Personal Chat Bot")

    # Embeddings and Retrieval Store
    embeddings = CohereEmbeddings(
        model="multilingual-22-12",
        cohere_api_key=cohere_api_key,
        user_agent="langchain",
    )
    store = Qdrant.from_documents(
        pages,
        embeddings,
        location=":memory:",
        collection_name="my_documents",
        distance_func="Dot",
    )

    def process_input():
        question = st.session_state["input_question"]
        prompt.append({"role": "user", "content": question})

        with messages_container:
            user_message(question)
            botmsg = bot_message("...", bot_name="Multilingual Personal Chat Bot")

        qa = RetrievalQA.from_chain_type(
            llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
            chain_type="stuff",
            retriever=store.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )

        answer = qa({"query": question})
        result = answer["result"].replace("\n", "").replace("Answer:", "")

        with st.spinner("Loading response .."):
            botmsg.update(result)

        # Add assistant's response to the prompt history
        prompt.append({"role": "assistant", "content": result})
        st.session_state["input_question"] = ""

    messages_container = st.container()
    st.text_input(
        "",
        placeholder="Type your message here",
        label_visibility="collapsed",
        key="input_question",
        on_change=process_input,
    )

    # Save the chat history in session state
    st.session_state["prompt"] = prompt
else:
    st.session_state["prompt"] = None
    st.warning("Upload a file to chat!")
