private ChatGPT is a very powerful concept and can be very useful to create a personal 
knowledgebase.  It can churn out information and answer any question, but the response was
 very slow to be of any practical use. Maybe that is because it is purely python-based ?

Thanks to the original author who created it.

I added a pdf file and queried and got this response after 10 minutes ....

Enter a query: Explain HybridAssetAllocation and its predecessor Bold Asset Allocation.
Hopefully at some point the speed> Answer:
 Here's an example of how you can create the HAACS function in Python using NumPy and Panda
ny different ways depending on your specific requirements or personal preferences as a deve
:```pythonimport pandas as pd  # for data manipulation (e.g., reading/writing csv) and savi
split a column into multiple columns by default using pivot table modedef hybrid_asset(star
        # define variables for dataframes (one per period) and labels/indexes of the assets
(periods=(len("BDSKY")+9))  # ASKT, BSPT - one per period for BDSKY (except first)
        # define variables to store the dataframes      df_list=[]```

> source_documents/HybridAssetAllocation.pdf:
Our new Hybrid Asset Allocation (HAA) is a follow-up on our Bold Asset Allocation (BAA). Us
as our inspiration, we try to compose a much simpler strategy for retail investors. We aim
balanced but aggressive strategy and much lower cash-fractions than BAA. An important role
new strategy is reserved for our ‘canary’ approach for crash protection, but now combined w
traditional dual momentum to arrive at our novel ‘hybrid’ approach. For HAA, we use a new s

> source_documents/HybridAssetAllocation.pdf:
Our Bold Asset Allocation (BAA) came in two variants: a balanced one (with Top6 out of a 12
risky universe) and an aggressive one (with Top1 out of a 4-asset risky universe). Our new
Asset Allocation (HAA) strategy aims at just one (balanced and aggressive) model with a Top
8-asset risky universe, using a ‘hybrid’ combination of traditional dual momentum and a new

> source_documents/HybridAssetAllocation.pdf:
By the ‘hybrid’ combination of traditional dual momentum with a new canary asset TIP for cr
protection, we are able to arrive at a very simple strategy called ‘Hybrid Asset Allocation
which is much simpler than our complex BAA (Bold Asset Allocation) model.

The recipe for this new HAA strategy is simple: on the close of the last trading day of eac

> source_documents/HybridAssetAllocation.pdf:
Keller, W.J., and A. Butler, 2015a, Elastic Asset Allocation (EAA), SSRN 2543979

Keller, W.J., A. Butler, and I. Kipnis, 2015b, Momentum and Markowitz (CAA), SSRN 2606884
 may improve so that we can put it for personal and career development to add richness in our lives.

# privateGPT
Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point. You can ingest documents and ask questions without an internet connection!

Built with [LangChain](https://github.com/hwchase17/langchain), [GPT4All](https://github.com/nomic-ai/gpt4all), [LlamaCpp](https://github.com/ggerganov/llama.cpp), [Chroma](https://www.trychroma.com/) and [SentenceTransformers](https://www.sbert.net/).

<img width="902" alt="demo" src="https://user-images.githubusercontent.com/721666/236942256-985801c9-25b9-48ef-80be-3acbb4575164.png">

# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

```shell
pip3 install -r requirements.txt
```

Then, download the LLM model and place it in a directory of your choice:
- LLM: default to [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin). If you prefer a different GPT4All-J compatible model, just download it and reference it in your `.env` file.

Rename `example.env` to `.env` and edit the variables appropriately.
```
MODEL_TYPE: supports LlamaCpp or GPT4All
PERSIST_DIRECTORY: is the folder you want your vectorstore in
MODEL_PATH: Path to your GPT4All or LlamaCpp supported LLM
MODEL_N_CTX: Maximum token limit for the LLM model
EMBEDDINGS_MODEL_NAME: SentenceTransformers embeddings model name (see https://www.sbert.net/docs/pretrained_models.html)
TARGET_SOURCE_CHUNKS: The amount of chunks (sources) that will be used to answer a question
```

Note: because of the way `langchain` loads the `SentenceTransformers` embeddings, the first time you run the script it will require internet connection to download the embeddings model itself.

## Test dataset
This repo uses a [state of the union transcript](https://github.com/imartinez/privateGPT/blob/main/source_documents/state_of_the_union.txt) as an example.

## Instructions for ingesting your own dataset

Put any and all your files into the `source_documents` directory

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),

Run the following command to ingest all the data.

```shell
python ingest.py
```

Output should look like this:

```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Loaded 1 new documents from source_documents
Split into 90 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run privateGPT.py to query your documents
```

It will create a `db` folder containing the local vectorstore. Will take 20-30 seconds per document, depending on the size of the document.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `db` folder.

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection, except for the first time you run the ingest script, when the embeddings model is downloaded.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python privateGPT.py
```

And wait for the script to require your input.

```plaintext
> Enter a query:
```

Hit enter. You'll need to wait 20-30 seconds (depending on your machine) while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: you could turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.


### CLI
The script also supports optional command-line arguments to modify its behavior. You can see a full list of these arguments by running the command ```python privateGPT.py --help``` in your terminal.


# How does it work?
Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `HuggingFaceEmbeddings` (`SentenceTransformers`). It then stores the result in a local vector database using `Chroma` vector store.
- `privateGPT.py` uses a local LLM based on `GPT4All-J` or `LlamaCpp` to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
- `GPT4All-J` wrapper was introduced in LangChain 0.0.162.

# System Requirements

## Python Version
To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11
To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   * Universal Windows Platform development
   * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the `gcc` component.

## Mac Running Intel
When running a Mac with Intel hardware (not M1), you may run into _clang: error: the clang compiler does not support '-march=native'_ during pip install.

If so set your archflags during pip install. eg: _ARCHFLAGS="-arch x86_64" pip3 install -r requirements.txt_

# Disclaimer
This is a test project to validate the feasibility of a fully private solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. The models selection is not optimized for performance, but for privacy; but it is possible to use different models and vectorstores to improve performance.
