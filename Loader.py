from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import os
import tempfile
from time import sleep


model=ChatOpenAI(
    api_key = st.secrets["OPENAI_API_KEY"]
)
parser=StrOutputParser()

if "vector_db" not in st.session_state:
    st.session_state.vector_db=None

if "messages" not in st.session_state:
    st.session_state.messages=[]

def document_flow(file):
    #doc_loader
    if file:
        file_extension=(
            os.path.splitext(
                file.name
            )
        )[1]

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as tmp_file:

            tmp_file.write(file.read())
            temp_path = tmp_file.name

        if file_extension=='.pdf':
            loader=PyPDFLoader(temp_path)
            docs=loader.load()
        elif file_extension=='.txt':
            loader=TextLoader(temp_path,encoding='utf-8')
            docs=loader.load()
        
        #splitter
        splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
        )

        chunks=splitter.split_documents(docs)
        
        #embeddings and vector_store
        embeddings=OpenAIEmbeddings()
        vector_db=FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
        )
        st.session_state.vector_db=vector_db
        st.session_state.document_uploaded=True



st.header("Document uploader Q&A chatbot")
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded=False
if not st.session_state.document_uploaded:
    file=st.file_uploader(
        "Upload document either .pdf or .txt format only",
        type=["pdf","txt"]
    )
    if file:
        with st.spinner("Processing....."):
            document_flow(file)
        st.markdown("Document Processed Successfully")
        sleep(2)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.vector_db:
    for messages in st.session_state.messages:
       with st.chat_message(messages['role']):
            st.write(messages['content']) 
    
    query=st.chat_input("Ask me anything?")
    with st.chat_message('user'):
        st.write(query)
    st.session_state.messages.append({'role':'user','content':query})
    if query:
        context=""
        multi_query_retriver=MultiQueryRetriever.from_llm(
            retriever=st.session_state.vector_db.as_retriever(
            search_kwargs={"k":3,"lambda_mult":0.5}
        ),
        llm=ChatOpenAI()
        )
        retrivers=multi_query_retriver.invoke(query)
        for i, doc in enumerate(retrivers):
            context+=doc.page_content+"\n\n"
        
        prompt=PromptTemplate(
        template="""You are a helpful assistant. Answer the {query} only from the provided context.
        1. If the user's message is a greeting, introduction, thanks, or casual conversation
        (e.g., "hi", "hello", "how are you"), respond naturally and DO NOT use retrieved context.
        2. Only use retrieved context when the question requires knowledge lookup.
        3. If retrieved context is irrelevant to the question, ignore it.
        4. Do not invent information.
        5. If information is unavailable, say:
        "I couldn't find relevant information.". If the context is insufficient,just say you dont know. {context}""",
        input_variables=['query','context']
        )

        chain=prompt | model | parser

        result=chain.invoke({'query':query,'context':context})
        with st.chat_message('ai'):
            st.write(result)
        st.session_state.messages.append({'role':'ai','content':result})
        
     



