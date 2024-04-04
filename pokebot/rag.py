import typing
import gradio as gr
from typing import List
import argparse
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



class GradioUserInference:
    @staticmethod
    def chat_interface_components(
            sample_func: typing.Callable,
            role_name: str,
    ):
        """
        The function `chat_interface_components` creates the components for a chat interface, including
        a chat history, message box, buttons for submitting, stopping, and clearing the conversation,
        and sliders for advanced options.
        """

        # _max_length = max_sequence_length
        # _max_new_tokens = max_new_tokens
        # _max_compile_tokens = max_compile_tokens

        with gr.Column("100%"):
            gr.Markdown(
                f"# <h1><center style='color:#6600FF;'>Pokebot: Damn Vulnerable RAG</center></h1> <h3><right style='color:#6600FF;'>{role_name}</right></h3>",
            )
            history = gr.Chatbot(
                elem_id="Pokebot",
                label="Pokebot",
                container=True,
                height="65vh",
            )
            prompt = gr.Textbox(
                show_label=False, placeholder='Enter Your Prompt Here.', container=False
            )
            with gr.Row():
                submit = gr.Button(
                    value="Run",
                    variant="primary"
                )
                stop = gr.Button(
                    value='Stop'
                )
                clear = gr.Button(
                    value='Clear Conversation'
                )
            with gr.Accordion(open=False, label="Advanced Options"):
                # system_prompt = gr.Textbox(
                #     value="",
                #     show_label=False,
                #     label="System Prompt",
                #     placeholder='System Prompt',
                #     container=False
                # )

                # temperature = gr.Slider(
                #     value=0.8,
                #     maximum=1,
                #     minimum=0.1,
                #     label='Temperature',
                #     step=0.01
                # )
                # top_p = gr.Slider(
                #     value=0.9,
                #     maximum=1,
                #     minimum=0.1,
                #     label='Top P',
                #     step=0.01
                # )
                # top_k = gr.Slider(
                #     value=50,
                #     maximum=100,
                #     minimum=1,
                #     label='Top K',
                #     step=1
                # )
                # repetition_penalty = gr.Slider(
                #     value=1.2,
                #     maximum=5,
                #     minimum=0.1,
                #     label='Repetition Penalty'
                # )
                # greedy = gr.Radio(
                #     value=True,
                #     label="Do Sample or Greedy Generation"
                # )

                mode = gr.Dropdown(
                    choices=["Chat", "Train", "Poison", "Unpoison"],
                    value="Chat",
                    label="Mode",
                    multiselect=False
                )
            gr.Markdown(
                "# <h5><center style='color:black;'>Powered by "
                "[Detoxio AI](https://detoxio.ai)</center></h5>",
            )

        inputs = [
            prompt,
            history,
            # system_prompt,
            mode,
            # greedy,
            # temperature,
            # top_p,
            # top_k,
            # repetition_penalty
        ]

        clear.click(fn=lambda: [], outputs=[history])
        sub_event = submit.click(
            fn=sample_func, inputs=inputs, outputs=[prompt, history]
        )
        txt_event = prompt.submit(
            fn=sample_func, inputs=inputs, outputs=[prompt, history]
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[txt_event, sub_event]
        )

    def handle_input(
            self,
            prompt: str,
            history: List[List[str]],
    ):
        raise NotImplementedError()

    def build_inference(
            self,
            sample_func: typing.Callable,
            role_name:str,
    ) -> gr.Blocks:
        """
        The function "build_inference" returns a gr.Blocks object that model
        interface components.
        :return: a gr.Blocks object.
        """
        with gr.Blocks() as block:
            self.chat_interface_components(
                sample_func=sample_func,
                role_name=role_name,
            )
        return block


class AssistantRole:
    def __init__(self, name, seed_urls, poison_files_pattern):
        self.name = name
        self.seed_urls = seed_urls
        self.poison_files_pattern = poison_files_pattern
        

class RAGApp(GradioUserInference):
    def __init__(self, assistant:AssistantRole):
        self._llm = ChatOpenAI()
        self._docs = []
        self._training_docs = []
        self.assistant = assistant
    
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
        if not pattern:
            pattern = self.assistant.poison_files_pattern
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
        data_folder = rel_path
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

    def handle_input(self,
            prompt: str,
            history: List[List[str]],
            mode: str,):
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
        response = handle(prompt, mode)
        history.append([prompt, ""])
        history[-1][-1] = response
        yield "", history
        # Initial update of documents and launch interface

    def run(self):
        print("Loading Initial Training Data...")
        if len(self.assistant.seed_urls) > 0:
            # Initialize with training data
            for url in self.assistant.seed_urls:
                self._add_website_url(url)
        else:
            # Just Initialize
            self._update_docs()
        demo = self.build_inference(self.handle_input, role_name=self.assistant.name)
        demo.launch(share=True)


