from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

@st.cache_resource
def load_bd():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder='./model_cache'
    )

    bd = Chroma(
        persist_directory="./chroma_bd_with_clean",
        embedding_function=embeddings
    )
    return bd

def find(query, bd):

    results = bd.similarity_search_with_score(query, k=1)

    return results

if __name__ == "__main__":

# common / common / 35807370 / 38981028 / 38981028 / 39122455 / 32804096
    questions = [
        "What is Alzheimer's disease?",
        "What is the difference between M1 and M2 microglial phenotypes?",
        "How does felodipine work as a potential AD treatment?",
        "What are the limitations of current felodipine administration?",
        "What role do cathepsins play in the pathogenesis of Alzheimer's disease?",
        "Why is the hippocampus the most affected brain structure in Alzheimer's disease?"
    ]

    for i, q in enumerate(questions):
        ans = find(q)
        print(f"{i + 1}. Question: {q}")
        print(f"Score: {ans[0][1]}")
        print(f"Answer:\n {ans[0][0].metadata['pmid']}\n{ans[0][0].page_content}")

