# Read on various lanchain modules here https://langchain.readthedocs.io/
# PyMuPDFLoader works well with windows character encoding

from langchain import PromptTemplate, OpenAI, LLMChain
import glob
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader


input_file_path = "./sources/employee_career_planning.pdf"


def initialize_vector_store(input_path, save=False):

    # check to see if path is a directory, if it is, load local vector database that may have been created already
    if os.path.isdir(input_path):
        faiss_file = glob.glob(input_path + '*/*.faiss')
        if faiss_file:
            vector_store = FAISS.load_local(input_path)
            return vector_store
        else:
            loader = DirectoryLoader(input_path)
    else:
        # otherwise load PDF file
        loader = PyMuPDFLoader(file_path=input_path)

    raw_document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    split_text = text_splitter.split_documents(raw_document)
    # test split --> print(split_text[0].page_content)

    # Now to vectorize split text into embeddings
    embeddings_tool = HuggingFaceHubEmbeddings()
    vector_store = FAISS.from_documents(split_text, embeddings_tool)

    if save:
        outputdirectory = os.path.split(input_path)[0]
        vector_store.save_local(outputdirectory)
    return vector_store


def get_similar_results(vector_database, query: str = None, num_of_chunks: int = 4):
    similar_docs = vector_database.similarity_search(query, num_of_chunks)
    results = []
    for s in similar_docs:
        results.append(f'{s.page_content}\n')
    print('\n\n'.join(results))
    return results

# Now all the code can be executed in two lines
# vector_store = initialize_vector_store('../path/file.pdf')
# similar_docs = get_similar_results(vector_store, "This is a query", 4)


def get_final_response():
    user_prompt = input("Enter your prompt: ")

    template = """You are an ai assistant developed by OpenAI for Mercedes-Benz. Your job is to make the lives of Mercedes-Benz employees easier. Use the following excerpt from a large document to respond to the user in a helpful way.
                        excerpt:{context}
                        user:{request}
                        response:
                        """
    prompt = PromptTemplate(template=template, input_variables=[
        "context", "request"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(
        temperature=0, max_tokens=256), verbose=True)
    vector_store = initialize_vector_store(input_file_path)
    # request = "What are the advantages of the career roadmap?"
    request = user_prompt
    context = get_similar_results(vector_store, request, 4)

    final_response = llm_chain.predict(context=context, request=request)
    print(final_response)
    return final_response


get_final_response()
