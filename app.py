from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title("Sidebar Chat pdf")
    st.markdown(
        '''
## About this app
This app is powered by Streamlit and is maintained by the Streamlit community.
'''
    )
    add_vertical_space(5)
    st.write("This is a test")
   
def main():
    st.header("Chat with pdf")
    load_dotenv()

    #upload pdf file
    pdf = st.file_uploader("Upload pdf file", type='pdf')   

    if pdf is not None:
        pd_reader = PdfReader(pdf)

        text=""
        for page in pd_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vector_store = pickle.load(f)
            st.write("Embeddings loaded from local disk")
            
        else:
                # EMBEDDINGS
            embeddings = OpenAIEmbeddings() # Load OpenAI embeddings

            vector_store = FAISS.from_texts(chunks,embedding=embeddings) # Create a vector store from the chunks
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vector_store,f)


            # Ask / accept user question or query
        query = st.text_input("Enter your query")
        st.write("Query:",query)

        if query:
            docs = vector_store.similarity_search(query=query)

            llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

        else:
            st.toast("No query entered",icon='❓')

            # st.write("Embeddings Computation completed")



if __name__ == "__main__":
    main()  