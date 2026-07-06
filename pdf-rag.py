#1.Ingest pdf files
#2. Extract text from pdf files and split into small chunks
#3. send the chunks to the embedding model
#4. save the embeddings in a vector database
#5. perform semilarity search on the vector database to find relevant chunks
#6.retreive the semilar documents and resent them to the user 
## run pip install -r requirements.txt to install the resuired packages


from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

#===== PDF INGESTION =====  
doc_path="C:\\Desktop\\ollama course\\data\\sensor_rei_prediction.pdf"
model="mistral"

#local pdf file uploads
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    documents = loader.load()
    print("done  loading the pdf file")
else :
    print("please provide a valid pdf file path")

#preview first page of the pdf file
content=documents[0].page_content
print(content[:500])
#====END PDF INGESTION =====

#===== EXTRACT TEXT AND SPLIT INTO SMALL CHUNKS ^__^ ===

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks=text_splitter.split_documents(documents)
print("done splitting")
print(f"number of chunks : {len(chunks)}")
print(f"first chunk : {chunks[0].page_content[:500]}")

#==== ADD TO VECTOR DATABASE =====
import ollama
ollama.pull("nomic-embed-text")

vector_db=Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple_rag"
)
print("done adding to vector database")

# RETRIEVAL
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

#set up our model to use 
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

# RAG PROMPT
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

#res=chain.invoke(input={"what is the main purpose of the sensor reid prediction system?"})

res = chain.invoke("how does the sensor reid prediction system work?")
print(res)
