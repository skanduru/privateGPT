# PrivateGPT App

This repository contains a FastAPI backend and queried on a commandline by curl. The PrivateGPT App provides an interface to privateGPT, with options to embed and retrieve documents using a language model and an embeddings-based retrieval system. All data remains can be local or private network. 


## Requirements

- Python 3.11 or later
- Minimum 16GB of memory

## Setup

1. Create a Python virtual environment using your preferred method.

2. Copy the environment variables from `example.env` to a new file named `.env`. Modify the values in the `.env` file to match your desired configuration. The variables to set are:
   - `PERSIST_DIRECTORY`: The directory where the app will persist data.
   - `MODEL_TYPE`: The type of the language model to use (e.g., "GPT4All", "LlamaCpp").
   - `MODEL_PATH`: The path to the language model file.
   - `EMBEDDINGS_MODEL_NAME`: The name of the embeddings model to use.
   - `MODEL_N_CTX`: The number of contexts to consider during model generation.
   - `API_BASE_URL`: The base API url for the FastAPI app, usually it's deployed to port:8000.


3. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the FastAPI Backend

To run the FastAPI backend, execute the following command:
```
gunicorn app:app -k uvicorn.workers.UvicornWorker --timeout 1500 &
```
OR
```
python app.py &
```
This command starts the backend server and automatically handles the necessary downloads for the language model and the embedding models. The `--timeout 500` option ensures that sufficient time is allowed for proper model downloading.

### Important Considerations

- Embedding documents is a quick process, but retrieval may take a long time due to the language model generation step. Optimization efforts are required to improve retrieval performance.

- The FastAPI backend can be used with any front-end framework of your choice. Feel free to integrate it with your preferred user interface.

- Community contributions are welcome! We encourage you to contribute to make this app more robust and enhance its capabilities.

The supported extensions for documents are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),

Certainly! Here are examples of how to call the API routes mentioned in the README:

### Root Route
- **Endpoint:** `GET /`
- **Description:** Get a simple greeting message to verify that the APIs are ready.
- **Example Usage:**
   ```bash
   curl -X GET http://localhost:8000/
   ```
   ```python
   import requests
   
   response = requests.get("http://localhost:8000/")
   print(response.json())
   ```

### Embed Route
- **Endpoint:** `POST /embed`
- **Description:** Embed files by uploading them to the server.
- **Example Usage:**
   ```bash
   curl -X POST -F "files=@file1.txt" -F "files=@file2.txt" -F "collection_name=my_collection" http://localhost:8000/embed
   ```
   ```python
   import requests
   
   files = [("files", open("file1.txt", "rb")), ("files", open("file2.txt", "rb"))]
   data = {"collection_name": "my_collection"}
   
   response = requests.post("http://localhost:8000/embed", files=files, data=data)
   print(response.json())
   ```

### Retrieve Route
- **Endpoint:** `POST /retrieve`
- **Description:** Retrieve documents based on a query.
- **Example Usage:**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"query": "sample query", "collection_name": "my_collection"}' http://localhost:8000/retrieve
   ```
   ```python
   import requests
   
   data = {"query": "sample query", "collection_name": "my_collection"}
   
   response = requests.post("http://localhost:8000/retrieve", json=data)
   print(response.json())
   ```

Please note that the actual URL (`http://localhost:8000/`) and the request payloads should be adjusted based on your specific setup and requirements.

### Sample session

```bash
gunicorn app:app -k uvicorn.workers.UvicornWorker --timeout 1500 &


 INFO:     Started server process [3281469]
 INFO:     Waiting for application startup.
 Loading documents from source_documents
 Loaded 1 documents from source_documents
 Split into 1 chunks of text (max. 500 characters each)
 2023-06-07 23:22:25.553046: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
 To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
 Using embedded DuckDB with persistence: data will be stored in: db
 Persist operation completed successfully.

 embeddings working
 File already exists.
 INFO:     Application startup complete.
 INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

```bash
  $ curl -X POST -F "files=@requirements.txt" -F "collection_name=my_collection" http://localhost:8000/embed
{"message":"Files embedded successfully\n","saved_files":["source_documents/requirements.txt"]}
```

```bash
  curl -X POST -H "Content-Type: application/json" -d '{"query": "curl", "collection_name": "my_collection"}' http://localhost:8000/retrieve
{"Question":"curl","Answer":" If you don't know the answer to this question or if there is another way that can be considered easier and quicker than sending a request in Python, I would recommend using cURL instead.","Documents":{"source_documents/README.md":"Embed Route\n\nEndpoint: POST /embed\n\nDescription: Embed files by uploading them to the server.\n\nExample Usage:\n   bash\n   curl -X POST -F \"files=@file1.txt\" -F \"files=@file2.txt\" -F \"collection_name=my_collection\" http://localhost:8000/embed\n   ```python\n   import requests\n\nfiles = [(\"files\", open(\"file1.txt\", \"rb\")), (\"files\", open(\"file2.txt\", \"rb\"))]\n   data = {\"collection_name\": \"my_collection\"}"}}
```
