from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline
import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyD6R6UQ85vU5oIpQZhP3eQSnyoTMebZ0WY")


model = genai.GenerativeModel("gemini-2.5-flash")

def answer_with_gemini(query, retriever):
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Answer the question strictly using the context below.
    If the answer is not present, say:
    "The answer is not available in the document."

    Context:
    {context}

    Question:
    {query}
    """
    print(context)
    print(query)
    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    print("📄 PDF QA Bot (Gemini Version Ready)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            break

        answer = answer_with_gemini(query, retriever)
        print("\n🧠 Answer:")
        print(answer)
        print("-" * 50)


# def load_rag_chain():
#     # Same embeddings as ingestion
#     embeddings = HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2"
#     )

#     vectorstore = FAISS.load_local(
#         "vectorstore",
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

#     # LLM (FREE)
#     pipe = pipeline(
#         "text2text-generation",
#         model="google/flan-t5-large",
#         max_new_tokens=256
#     )

#     llm = HuggingFacePipeline(pipeline=pipe)

#     prompt = ChatPromptTemplate.from_template(
#         """
#         Answer the question ONLY using the context below.
#         If the answer is not in the context, say "I don't know".

#         Context:
#         {context}

#         Question:
#         {question}
#         """
#     )

#     chain = (
#         {
#             "context": retriever,
#             "question": lambda x: x
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return chain


# if __name__ == "__main__":
#     chain = load_rag_chain()

#     print("📄 PDF QA Bot Ready (Modern LangChain)")
#     print("Type 'exit' to quit.\n")

#     while True:
#         query = input("Ask a question: ")
#         if query.lower() == "exit":
#             break

#         answer = chain.invoke(query)
#         print("\n🧠 Answer:")
#         print(answer)
#         print("-" * 50)
