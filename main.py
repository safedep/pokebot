
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_retrieval_chain
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader



class RAGApp:

    def __init__(self):
        self._llm = ChatOpenAI()
        self._docs = []
        self._training_docs = []

    def _add_website_url(self, url):
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        self._docs.extend(documents)
        self._training_docs.extend(documents)
        self._update_docs()
    
    def _poison(self, pattern):        
        loader = DirectoryLoader("./data/poisoning/", glob=f"**/*{pattern}*", loader_cls=TextLoader, show_progress=True)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        self._docs.extend(documents)
        self._update_docs()
        return documents

    def _get_med_qa_docs(self):
        return self._get_json_docs("data/medqa/small/")

    def _update_docs(self):
        embeddings = OpenAIEmbeddings()
        vector = FAISS.from_documents(self._docs, embeddings)
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}""")

        document_chain = create_stuff_documents_chain(self._llm, prompt)
        retriever = vector.as_retriever()
        self.retrieval_chain = create_retrieval_chain(retriever, document_chain)


    def run(self):
# response = retrieval_chain.invoke({"input": "How did someone discard role?"})
# print(response["answer"])
        def handle(text, mode):
            if mode.lower() == "train":
                self._add_website_url(text)
                return "Done"
            elif mode.lower() == "poison":
                docs = self._poison(text)
                print(docs)
                return "Done"
            elif mode.lower() == "unpoison":
                self._docs=self._training_docs
                self._update_docs()
            else:
                response = self.retrieval_chain.invoke({"input": text})
                return response["answer"]

        demo =  gr.Interface(
            fn=handle,
            inputs=["text", gr.Radio(['Chat', 'Train', 'Poison', "Unpoison"], value="Chat")],
            outputs=["text"],
        )

        demo.launch(share=True)



# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

# document_chain = create_stuff_documents_chain(llm, prompt)

# from langchain_core.output_parsers import StrOutputParser

# output_parser = StrOutputParser()

# chain = prompt | llm | output_parser

# chain.invoke({"input": "how can langsmith help with testing?"})

app = RAGApp()
app.run()
