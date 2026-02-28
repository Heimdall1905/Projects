from find_answer import find, load_bd
from openai import OpenAI
from api import api

import streamlit as st

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api,
)

prompt = """
You are an assistant who answers questions using only context

IMPORTANT RULES:
1. Use only information from the context provided.
2. If the context does not contain an answer to the question, say: There is no such information in the database.
3. DO NOT use your own or general knowledge.

"""

st.title("Alzheimer's Research Assistant")

bd = load_bd()

question = st.text_input("Введите ваш вопрос:")

if st.button("Спросить"):
    st.write("## Ответ:")
    if question:
        res = find(question, bd)
        content = res[0][0].page_content

        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-preview:free",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context: {content}\n\nQuestion: {question}"""}
            ]
        )

        st.write(response.choices[0].message.content)

        st.write("## Источники:")
        st.write(f"PMID: {res[0][0].metadata['pmid']}")
        st.write(f"Authors: {', '.join(res[0][0].metadata['authors'])}")

        st.write('## Ответ основан на:')
        st.write(content)

    else:
        st.write("Поле пустое, задайте вопрос")
