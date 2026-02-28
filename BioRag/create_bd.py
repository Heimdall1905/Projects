from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from tqdm import tqdm

from metapub import PubMedFetcher
from metapub import FindIt
fetch = PubMedFetcher()

import json, re

with open('dataset.json', 'r') as f:
  data = json.load(f) # загружаем подготовленный датасет

print(f"Всего статей в датасете: {len(data)}")
print(data[0].keys())

print("Пример темы: " + fetch.article_by_pmid(data[0]['pmid']).title + "\n") # проверка, что работает

CHUNK_SIZE = 500
OVER = 0.3

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_SIZE * OVER,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)

# очистка текста
def clean_text(text: str) -> str:
    text = re.sub(r'<!--.*?-->', '', text)
    text = re.sub(r'\[\d+(?:[-–,]\s*\d+)*\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[\d+(?:[-–,]\s*\d+)+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

docs = []
for i in tqdm(data):
  article = fetch.article_by_pmid(i['pmid'])
  abstract = article.abstract

  # простое объединение
  doc = Document(
      page_content= clean_text(abstract + i['introduction'] + i['conclusion']),
      metadata={
          'title': article.title,
          'pmid': i['pmid'],
          'authors': article.authors,
          'year': article.year,
          'url': FindIt(i['pmid']).url
      }
  )

  chunks = splitter.split_documents([doc])

  docs.extend(chunks)

max_length = 0
max_text = ''
for i in docs:
  j = len(i.page_content)
  if j > max_length:
      max_length = j
      max_text = i.page_content


print(f"\nСоздано {len(docs)} чанков из {len(data)} статей")
print(f"Самый длинный чанк: {len(max_text)}")
print(max_text)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

bd = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_bd_with_clean_2",
cache_folder='./model_cache'
)

