from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

model="mistral"
embedding_model="nomic-embed-text"
doc_path="C:\\Desktop\\ollama course\\data\\sensor_rei_prediction.pdf"


def ingest_pdf(doc_path, model):
   if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    documents = loader.load()
    print("done  loading the pdf file")
   else :
    print("please provide a valid pdf file path")
    exit()

   return documents

def split_pdf(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks=text_splitter.split_documents(documents)
    print("done splitting")
    return chunks

def add_to_vector_db(chunks, embedding_model):
  ollama.pull(embedding_model)
  vector_db=Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model=embedding_model),
    collection_name="simple_rag"
)
  print("done adding to vector database")
  return vector_db

def retrieve_from_vector_db(vector_db, model):
  llm=ChatOllama(model=model)
  QUERY_PROMPT=PromptTemplate(
    input_variables=["question"],
    template=""" You are an AI language model assistant . your task is to generate five 
    different versions of the givent user question to retreive relevant documents from 
    the vector database . By generating multiple versions of the question , your goal is 
    to help the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by new lines.
    Original question : {question},
    """ 
)
  retriever=MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm=llm,
    prompt=QUERY_PROMPT,
)
  return retriever,llm

def main():
  documents=ingest_pdf(doc_path, model)
  chunks=split_pdf(documents)
  vector_db=add_to_vector_db(chunks, embedding_model)
  retriever,llm=retrieve_from_vector_db(vector_db, model)
  template="""
            Answer the question based only on the following context
            {context}
            Question : {question}
        """

  prompt=ChatPromptTemplate.from_template(template)
  chain=(
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm 
    | StrOutputParser()
    )
  res = chain.invoke("how does the sensor reid prediction system work?")
  print(res)


if __name__ == "__main__":
    main()
