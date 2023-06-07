from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
import json

from ingest2 import load_documents, main as ingest_main
from privateGPT import query_results
from privateGPT import Job, getQnA

app = FastAPI()

# We store our Job instance globally so it can be accessed from anywhere in the app.
job_instance = None

@app.on_event("startup")
async def startup_event():
    global job_instance
    job_instance = Job(target=getQnA)
    ingest_main()
    job_instance.start()

@app.on_event("shutdown")
async def shutdown_event():
    global job_instance
    if job_instance:
        job_instance.shutdown()

class Item(BaseModel):
    key: str
    value: Optional[str] = None

data_store = []

async def model_download():
    """
    match model_type:
        case "LlamaCpp":
            url = "https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin"
        case "GPT4All":
            url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
    """
    if model_type == "LlamaCpp":
            url = "https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin"
    else:
            url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
    folder = "models"
    parsed_url = urllib.parse.urlparse(url)
    filename = os.path.join(folder, os.path.basename(parsed_url.path))
    # Check if the file already exists
    if os.path.exists(filename):
        print("File already exists.")
        return
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    # Run wget command to download the file
    os.system(f"wget {url} -O {filename}")
    global model_path 
    model_path = filename
    os.environ['MODEL_PATH'] = filename
    print("model downloaded")
# Example route
@app.get("/")
async def root():
    return {"message": "Hello, the APIs are now ready for your embeds and queries!"}

@app.post("/ingest", status_code=201)
async def ingest_data(item: Item):
    data_store.append(item.dict())
    if item.value is None:
        raise HTTPException(status_code=404, detail="source file or directory not specified")
    load_documents(item.value)
    return {'msg': 'Data ingested successfully'}

@app.post("/query")
async def query_data(item: Item):
    query = item.value
    if query is None:
        raise HTTPException(status_code=404, detail="Item not found")
    results = query_results(query)
    if results is None:
        raise HTTPException(status_code=404, detail="No results found")
    payload = {}
    payload["query"] = results["query"]
    payload["answer"] = results["result"]
    for doc in results['source_documents']:
        payload[doc.metadata['source']] = doc.page_content
    """
    answer, doc_list = results
    num_docs_per_payload = 10
    initial_payload = {
        "query": query,
        "answer": answer,
        "docs": doc_list[:num_docs_per_payload]
    }
    """
    return json.dumps(payload)
    # return initial_payload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
