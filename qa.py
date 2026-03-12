from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from transformers import pipeline
import google.genai as genai
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

def answer_with_gemini(query, retriever):
    try:
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
        
        # Debug prints
        print("\n[DEBUG] Context retrieved:")
        print(context[:200] + "...")  # Print first 200 chars
        print("\n[DEBUG] Query:")
        print(query)
        
        # Generate response using NEW API syntax
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Debug the response object
        print(f"\n[DEBUG] Response type: {type(response)}")
        
        # Extract answer
        answer = response.text
        
        print(f"\n[DEBUG] Answer generated successfully\n")
        
        return answer
        
    except Exception as e:
        print(f"\n❌ Error in answer_with_gemini: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return "Error generating response"


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
    print(f"API Key loaded: {os.getenv('GENAI_API_KEY')[:10]}...")
    
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            break

        answer = answer_with_gemini(query, retriever)
        print("\n🧠 Answer:")
        print(answer)
        print("-" * 50)
