
from dotenv import load_dotenv
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
from fastapi import FastAPI, UploadFile, File
from typing import List, Optional
import shutil
import pandas as pd
load_dotenv()

embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME') or 'distilbert-base-uncased'
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = 'models/orca-mini-3b-gguf2-q4_0.gguf'
model_n_ctx = os.environ.get('MODEL_N_CTX')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
collection_name = os.environ.get('COLLECTION_NAME', 'collection')

from constants import CHROMA_SETTINGS

secret = ''
st.set_page_config(
    page_title="Questify",
    page_icon=":robot:"
)



st.session_state['Bot_msg'] = []
st.session_state['History_msg'] = []

def private_gpt_generate_msg(human_msg):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory,collection_name=collection_name, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    # 
    if model_type == 'LlamaCpp':
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
    elif model_type == 'GPT4All':
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    else:
        print(f"Model {model_type} not supported!")
        return
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    # Get the answer from the chain
    res = qa(human_msg)
    # print(res)   
    answer, docs = res['result'], res['source_documents']
    return answer
	
def Bot_generate_msg(human_msg):
    return private_gpt_generate_msg(human_msg)

st.header("Questify")
st.write("Welcome to Questify! Ask me anything!")


# Sidebar
with st.sidebar:

    st.header("Upload your own documents")
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(source_directory, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success("File(s) uploaded successfully")
    st.sidebar.info(model_type)
    st.sidebar.info(model_path)
    st.sidebar.info(source_directory)
    st.sidebar.info(collection_name)
    
    
with st.expander("View source directory"):
    files = os.listdir(source_directory)
    size = [f"{os.path.getsize(os.path.join(source_directory, file))} bytes" for file in files]
    st.dataframe(pd.DataFrame({"File": files, "Size (bytes)": size}), use_container_width=True, hide_index=True)


# if 'Bot_msg' not in st.session_state:
#     st.session_state['Bot_msg'] = []

# if 'History_msg' not in st.session_state:
#     st.session_state['History_msg'] = []


def get_text():
    prompt = st.chat_input("Say something")
    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")
    return prompt


user_input = get_text()

if user_input:
    st.session_state.History_msg.append(user_input)
    with st.spinner("Thinking... (please upgrade to 16 gb ram)"):
        st.session_state.Bot_msg.append(Bot_generate_msg(user_input))
    

if st.session_state['Bot_msg']:
    for i in range(len(st.session_state['Bot_msg'])-1, -1, -1):
        with st.chat_message("assistant"):
            st.markdown(f"Bot: {st.session_state['Bot_msg'][i]}")
        with st.chat_message("user"):
            st.markdown(f"User: {st.session_state['History_msg'][i]}")


# clear chat
if st.sidebar.button("Clear Chat"):
    st.session_state['Bot_msg'] = []
    st.session_state['History_msg'] = []
    st.rerun()

# display content in source directory

 
        