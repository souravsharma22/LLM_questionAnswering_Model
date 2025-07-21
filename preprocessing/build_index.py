import os, json
import faiss
import numpy as np
from newspaper import Article
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

urls = [
    "https://restthecase.com/knowledge-bank/indian-political-scams",
    "https://www.legalserviceindia.com/legal/article-12888-political-scams-in-india.html",
    "https://blog.ipleaders.in/10-biggest-indian-political-scams/",
    "https://en.wikipedia.org/wiki/Politics_of_India"
]

pdf_paths = [
    # "data/Indian_Politics_and_Society_Since_Independence_Bidyut_Chakrabarty.pdf",
    "data/INDIAN_GOVT__POLITICS.pdf",
    "data/IPG_const.pdf"
]
#getting data from the websites
def fetch_url_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text
#define getting data from the pdfs
def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

#collecting all the text from all the website 
all_text = []
for url in urls:
    print(f"Fetching data from {url}")
    all_text.append(fetch_url_text(url))
for path in pdf_paths:
    print(f"Extracting data from book : {path}")
    all_text.append(extract_pdf_text(path))

#splitting data into chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap = 50)
chunks = []
for text in all_text:
    chunks.extend(splitter.split_text(text))

#Embedding 

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

#saving these data
os.makedirs("artifacts", exist_ok=True)
faiss.write_index(index, 'artifacts/rag_index.faiss')

with open("artifacts/rag_chunks.json", "w") as f:
    json.dump(chunks, f)

#saving model

model.save("artifacts/rag_embedder")

print("Preprocessing completed")