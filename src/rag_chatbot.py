!pip install unstructured
!pip install sentence-transformers
!pip install chromadb --use-deprecated=legacy-resolver
!pip install openai
!pip install langchain --use-deprecated=legacy-resolver
!pip install langchain-community langchain-core

from langchain.document_loaders import TextLoader
documents = TextLoader("./data/AI.txt").load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

# split docs to chunk
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# save splitted docs to `docs` variables
docs = split_docs(documents)

from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# save vector to Chromdb
from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings)

import os
# os.environ["OPENAI_API_KEY"] = "###"

from langchain.chat_models import ChatOpenAI
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

# getting answers for query using qna chains
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# custom query
# perform similarity search
query = "What is AI?"
matching_docs = db.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
answer
