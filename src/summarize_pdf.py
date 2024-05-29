import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

# split texts to chunks using CharacterTextSplitter
def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # emdedding process(converting vector using HuggingFaceEmbeddings model)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

# create website using streamlit
def main():
    st.title("Summarize into a PDF.")
    st.divider()
    try:
        # os.environ["OPEN_API_KEY"] = "###"
    except ValueError as e:
        st.error(str(e))
        return

    pdf = st.file_uploader("Upload PDF file.", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
    
        for page in pdf_reader.pages:
            text += page.extract_text()
    
        documents = process_text(text)
        query = "Summarize the content of the uploaded PDF file into approximately 3-5 sentences."

        if query:
            docs = documents.similarity_search(query)
            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)
            chain - load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
            st.subheader('===result===')
            st.write(response)

if __name__ == '__main__':
    main()