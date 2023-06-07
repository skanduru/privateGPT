#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import sys
import threading
import time
import pdb

sleep_interval = 1

load_dotenv()

from enum import Enum

class appMode(Enum):
    fastAPI = 1
    standalone = 2

myappmode = appMode.standalone
hide_source = False

class Job(threading.Thread):
    def __init__(self, target, args = ()):
        if args:
            args = (self,) +  args
        else:
            args = ()

        super(Job, self).__init__(target = target, args = args)
        self.shutdown_flag = threading.Event()

    def wait_until_qa_initialized(self):
        while qa is None:
            time.sleep(0.1)

    def start(self):
        super(Job, self).start()

    def ok_to_run(self):
        return not self.shutdown_flag.is_set()

    def shutdown(self):
        self.shutdown_flag.set()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

qa = None

def getQnA():
    global qa
    # Parse the command line arguments
    args = parse_arguments()
    hide_source = args.hide_source

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    if sys.version_info.minor == 8:
        # Define a dictionary mapping model types to functions that create the models.
        model_types = {
            "LlamaCpp": lambda: LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False),
            "GPT4All": lambda: GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False),
        }
        
        # Try to get the function from the dictionary.
        model_func = model_types.get(model_type)
        
        # If the function exists, call it to create the model. Otherwise, print an error and exit.
        if model_func:
            llm = model_func()
        else:
            print(f"Model {model_type} not supported!")
            exit()
    else:
        """
        Uncomment when using 3.10
        match model_type:
            case "LlamaCpp":
                llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
            case "GPT4All":
                llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
            case _default:
                print(f"Model {model_type} not supported!")
                exit;
        """
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

    # while True:
    #    time.sleep(sleep_interval)


def query_results(query: str):
    # Wait until 'qa' is not None
    while qa is None:
        time.sleep(0.1)
    # Get the answer from the chain
    res = qa(query)
    answer, docs = res['result'], [] if hide_source else res['source_documents']
    return answer, docs

def query_sync():
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [] if hide_source else res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    agent = Job(target = getQnA, args = ())
    agent.start()
    if myappmode == appMode.standalone:
        query_sync()
