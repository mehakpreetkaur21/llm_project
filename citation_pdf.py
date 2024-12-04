from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import os
from langchain.schema import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr

def process_pdf(files, query, output_dir="./pdf_pages"):
    if not files:
        return "Please upload valid PDF files.", []

    text = ""
    page_map = {}  # To map text chunks to page numbers
    all_pages = []  # Store page image paths

    # Step 1: Extract text and map to pages
    for file_index, file in enumerate(files):
        pdf_reader = PdfReader(file)
        pages = convert_from_path(file, dpi=200, output_folder=output_dir, fmt="png")
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""
            text += page_text
            # Save page image path
            page_image_path = os.path.join(output_dir, f"file_{file_index}_page_{page_number}.png")
            all_pages.append(page_image_path)
            # Map extracted text to page number
            page_map[len(text)] = (file_index, page_number)

    # Step 2: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # Step 3: Create embeddings and vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    documents = [
        Document(page_content=chunk, metadata={"page_number": i + 1})
        for i, chunk in enumerate(chunks)
    ]
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name='local-rag'
    )

    # Step 4: Retrieve and generate answer
    local_model = 'llama3.2:1b'
    llm = ChatOllama(model=local_model)
    query_prompt = PromptTemplate(
        input_variables=['question'],
        template='''You are an AI assistant. Generate five alternative questions for better retrieval
        from a vector database based on the user's query. Separate alternatives with newlines.
        Original question: {question}'''
    )

    retriever = MultiQueryRetriever.from_llm(vectordb.as_retriever(), llm, prompt=query_prompt)

    template = '''Answer the question based ONLY on the following context:
    {context}
    Question: {question}'''

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"question": query})

    # Step 5: Find relevant pages from results
    relevant_pages = []
    for doc in retriever.retrieve(query):
        metadata = doc.metadata
        page_number = metadata.get("page_number")
        if page_number:
            relevant_pages.append(all_pages[page_number - 1])

    # Return answer and relevant images
    return result, relevant_pages


# from pdf2image import convert_from_path
# import os
# import gradio as gr
# def pdf_to_image(pdf_path,output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     pages=convert_from_path(pdf_path,dpi=300)
#     image_paths=[]
#     for i ,page in enumerate(pages):
#         image_path=os.path.join(output_dir,f"page_{i+1}.png")
#         page.save(image_path,"PNG")
#         image_paths.append(image_path)
#     return image_paths


# def query_pdf(query,pdf_path):
#     relevant_pages=[1,3]
#     answers="Extracted answers for query"
#     images=[f"./output_images/page_{p}.png" for p in relevant_pages]
#     return answers,images

pdf_file=gr.File(label="Upload_pdf",file_types=[".pdf"])
query_input=gr.Textbox(label="Enter your query")
output_text = gr.Textbox(label="Answer")
output_images=gr.Gallery(label="Citations")
gr.Interface(
    fn=process_pdf,
    inputs=[query_input,pdf_file],
    outputs=[output_text,output_images],
    title="PDF query with images"
).launch(share=True,debug=True)