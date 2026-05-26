from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import streamlit as st
import fitz
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

# to check the if the uploaded pdf is normal pdf(text-based) or scanned pdf
def is_scanned_pdf(uploaded_file):
    uploaded_file.seek(0)
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    for page in pdf:
        text = page.get_text().strip()

        # If any page contains text → treat as normal PDF
        if text:
            return False

    # No text found in all pages → likely scanned PDF
    return True


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
            if not is_scanned_pdf(file):
                loader=PyPDFLoader(temp_path)
            else:
                loader=UnstructuredPDFLoader(temp_path)
            
        elif file_extension=='.txt':
            loader=TextLoader(temp_path,encoding='utf-8')
            
        elif file_extension =='.docx':
            loader=Docx2txtLoader(temp_path)
            
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
        "Upload document",
        type=["pdf","txt","docx"]
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
    
    if query:
        st.session_state.messages.append({'role':'user','content':query})
        with st.chat_message('user'):
            st.write(query)
        context=""

        # Vector retriever using MMR
        base_retriever = st.session_state.vector_db.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5,
                    }
                )

        multi_query_retriver=MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=ChatOpenAI()
        )

        
        retrivers=multi_query_retriver.invoke(query)

        for i, doc in enumerate(retrivers):
            context+=doc.page_content+"\n\n"
        
        prompt=PromptTemplate(
        template="""You are a helpful assistant. Answer the query only from the provided context.
        1. If the user's message is a greeting, introduction, thanks, or casual conversation
        (e.g., "hi", "hello", "how are you"), respond naturally and DO NOT use retrieved context.
        2. Only use retrieved context when the question requires knowledge lookup.
        3. If retrieved context is irrelevant to the question, ignore it.
        4. Do not invent information.
        5. If information is unavailable, say:
        "I couldn't find relevant information.". If the context is insufficient,just say you dont know.
        6.if the user's message is to summarize,what is this pdf about
        (e.g.., "summarize","summary","what is this pdf about","overview","key points") then
        answer accordingly dont give me same answers for each type of these questions respond naturally like a summarizer pdf
        Read all retrieved sections and produce:

            1. Main topic
            2. Purpose of the document
            3 . Key sections
            4. Important findings
            5. Short TL;DR



          context: {context}
          query: {query}""",
        input_variables=['query','context']
        )

        chain=prompt | model | parser

        result=chain.invoke({'query':query,'context':context})
        st.session_state.messages.append({'role':'ai','content':result})
        with st.chat_message('ai'):
            st.write(result)
        
        
     



