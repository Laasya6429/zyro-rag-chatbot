import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro Dynamics HR Assistant", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about Zyro Dynamics HR policies")

@st.cache_resource
def initialize_rag():
    loader = PyPDFDirectoryLoader("docs/")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    return retriever, llm

retriever, llm = initialize_rag()

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR assistant for Zyro Dynamics. Answer using only the provided context.
If the answer is not in the context, say "I don't have information about that in our HR policies."

Context: {context}
Question: {question}
Answer:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
Is this question related to HR topics like leave, benefits, work from home, 
code of conduct, performance, compensation, IT security, onboarding, or travel expenses?
Respond with only YES or NO.

Question: {question}
""")

REFUSAL_MESSAGE = "I'm only able to answer questions related to Zyro Dynamics HR policies."

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_bot(question):
    oos_check = OOS_PROMPT | llm | StrOutputParser()
    classification = oos_check.invoke({"question": question}).strip().upper()

    if "NO" in classification:
        return REFUSAL_MESSAGE, []

    docs = retriever.invoke(question)
    context = format_docs(docs)
    prompt = RAG_PROMPT.invoke({"context": context, "question": question})
    response = llm.invoke(prompt)
    answer = StrOutputParser().invoke(response)

    sources = list(set([
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in docs
    ]))

    return answer, sources


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_bot(prompt)
        st.markdown(answer)
        if sources:
            st.caption("📄 Sources: " + " | ".join(sources))

    full_response = answer
    if sources:
        full_response += f"\n\n📄 *Sources: {' | '.join(sources)}*"

    st.session_state.messages.append({"role": "assistant", "content": full_response})