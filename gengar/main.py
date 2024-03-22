import argparse
import gradio as gr
import pkg_resources
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader


class RAGApp:
    def __init__(self, seed_training_urls=None):
        self._llm = ChatOpenAI()
        self._docs = []
        self._training_docs = []
        self._seed_training_urls = seed_training_urls or []

    def _add_website_url(self, url):
        # Load documents from the web
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        # Extend document lists and update
        self._docs.extend(documents)
        self._training_docs.extend(documents)
        self._update_docs()

    def _poison(self, pattern):
        # Load poisoned documents from directory
        loader = DirectoryLoader(self._get_data_folder_location(rel_path="./data/poisoning/"), 
                                 glob=f"**/*{pattern}*", loader_cls=TextLoader, show_progress=True)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        # Extend document lists and update
        self._docs.extend(documents)
        self._update_docs()
        return documents

    def _get_data_folder_location(self, rel_path):
        # Get absolute path to data folder
        data_folder = pkg_resources.resource_filename(__name__, rel_path)
        return data_folder

    def _update_docs(self):
        # Update embeddings and create retrieval chain
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
        def handle(text, mode):
            # Handle different modes
            if mode.lower() == "train":
                self._add_website_url(text)
                return "Done"
            elif mode.lower() == "poison":
                docs = self._poison(text)
                return "Done"
            elif mode.lower() == "unpoison":
                # Reset to training documents
                self._docs = self._training_docs
                self._update_docs()
                return "Done"
            else:
                # Use retrieval chain for answering questions
                response = self.retrieval_chain.invoke({"input": text})
                return response["answer"]

        # Interface for interaction
        demo = gr.Interface(
            fn=handle,
            inputs=["text", gr.Radio(['Chat', 'Train', 'Poison', "Unpoison"], value="Chat")],
            outputs=["text"],
        )

        print("Loading Initial Training Data...")
        if len(self._seed_training_urls) > 0:
            # Initialize with training data
            for url in self._seed_training_urls:
                self._add_website_url(url)
        else:
            # Just Initialize
            self._update_docs()

        # Initial update of documents and launch interface
        demo.launch(share=True)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RAG App with command-line options.')
    parser.add_argument('--seed', type=str, help='Path to a file containing URLs for seeding training documents.')
    args = parser.parse_args()

    # If seed is provided, parse and load the URLs in an array
    seed_list = []
    if args.seed:
        seed_file = pkg_resources.resource_filename(__name__, args.seed)
        with open(seed_file, 'r') as file:
            seed_list = [line.strip() for line in file.readlines()]

    # Initialize and run the RAGApp
    app = RAGApp(seed_training_urls=seed_list)
    app.run()

if __name__ == "__main__":
   main()
