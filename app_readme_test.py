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
    st.title("Visualize your Markdown file")
    add_vertical_space(2)
   
def main():
    st.header("Chat with pdf")
    load_dotenv()

    #upload Markdown file
    md_file = st.file_uploader("Upload markdown file", type='md')
    if md_file is not None:
        content = md_file.read()
        # st.write(content)
        st.sidebar.markdown(content.decode())


    if md_file is not None:
        text = md_file.read()
        st.write("Text:",text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        st.write("Chunks:",chunks)
        

        store_name = md_file.name[:-3]

        if os.path.exists(f"{store_name}.pkl") and os.path.getsize(f"{store_name}.pkl") > 0:
            with open(f"{store_name}.pkl","rb") as f:
                vector_store = pickle.load(f)
            st.write("Embeddings loaded from local disk")
            
        else:
            # EMBEDDINGS
            embeddings = OpenAIEmbeddings() # Load OpenAI embeddings

            if chunks:
                vector_store = FAISS.from_texts(chunks,embedding=embeddings) # Create a vector store from the chunks
            else:
                st.write("No valid data to generate embeddings.")
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


if __name__ == "__main__":
    main()  