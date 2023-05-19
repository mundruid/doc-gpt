#!/usr/bin/env python
# dotenv is a library that allows us to securely load env variables
from dotenv import load_dotenv

# used to load an individual file (TextLoader) or multiple files (DirectoryLoader)
from langchain.document_loaders import TextLoader, DirectoryLoader

# used to split the text within documents and chunk the data
from langchain.text_splitter import CharacterTextSplitter

# use embedding from OpenAI (but others available)
from langchain.embeddings import OpenAIEmbeddings

# using Chroma database to store our vector embeddings
from langchain.vectorstores import Chroma

# use this to configure the Chroma database
from chromadb.config import Settings

# we'll use the chain that allows Question and Answering and provides source of where it got the data from. This is useful if you have multiple files. If you don't need the source, you can use RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

# we'll use the OpenAI Chat model to interact with the embeddings. This is the model that allows us to query in a similar way to ChatGPT
from langchain.chat_models import ChatOpenAI

# we'll need this for reading/storing from directories
import os

# finds .env file and loads the vars
load_dotenv()

__file__ = "doc_gpt.ipynb"
# get absolute path
FULL_PATH = os.path.dirname(os.path.abspath(__file__))
CURRENT_PATH = FULL_PATH.split("doc_gpt")[0]

# get the path of the db with docs
DB_DIR = os.path.join(CURRENT_PATH, "db")

# this can change to a path with files such as publications, meeting minutes etc.
DATA_DIR = os.path.join(CURRENT_PATH, "data")

# load individual files
doc_loader = TextLoader(f"{DATA_DIR}/MSFT_Call_Transcript.txt", encoding="utf8")

# use directory loader for dirs
# doc_loader = DirectoryLoader(DB_DIR)

# load the document
document = doc_loader.load()

# get a splitter with relevant parameters
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)

# split the doc data
split_docs = text_splitter.split_documents(document)

# load embeddings from OpenAI
openai_embeddings = OpenAIEmbeddings()

# configure DB
client_settings = Settings(
    chroma_db_impl="duckdb+parquet",  # store parquet files/DuckDB
    persist_directory=DB_DIR,
    anonymized_telemetry=False,
)

# create class level var vector store
vector_store = None

# check if db exists already
# if not, create ig
if not os.path.exists(DB_DIR):
    vector_store = Chroma.from_documents(
        split_docs,
        openai_embeddings,
        persist_directory=DB_DIR,
        client_settings=client_settings,
        collection_name="transcript_store",
    )
    vector_store.persist()
else:
    vector_store = Chroma(
        collection_name="transcript_store",
        persist_directory=DB_DIR,
        embedding_function=openai_embeddings,
        client_settings=client_settings,
    )

# create and configure our chain
# we're using ChatOpenAI LLM with the 'gpt-3.5-turbo' model
# we're setting the temperature to 0. The higher the temperature, the more 'creative' the answers. In my case, I want as factual and direct from source info as possible
# 'stuff' is the default chain_type which means it uses all the data from the document
# set the retriever to be our embeddings database
qa_with_source = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
)


def query_document(question):
    return qa_with_source({"question": question})


while True:
    print("What is your query? ", end="")
    user_query: str = input("\033[33m")
    print("\033[0m")
    if user_query == "quit":
        break
    response: dict[str, str] = query_document(user_query)
    # make the answer green and source blue using ANSI codes
    print(f'Answer: \033[32m{response["answer"]}\033[0m')
    print(f'\033[34mSources: {response["sources"]}\033[0m')
