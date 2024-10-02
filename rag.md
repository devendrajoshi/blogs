
# Building a RAG-Powered API with FastAPI and OllamaLLM

In the ever-evolving field of artificial intelligence, **Retrieval Augmented Generation (RAG)** is a standout concept that uniquely combines information retrieval and language generation. This blog post will explore the details of RAG and its practical implementation using FastAPI and OllamaLLM.

RAG is an innovative approach that merges the capabilities of **Large Language Models (LLMs)** and **document retrieval** systems to produce detailed and contextually appropriate responses. It works by fetching relevant documents or passages from a vast text corpus and then uses these retrieved texts as the context for generating a response.

The advantages of using RAG are significant, especially for tasks like question answering and summarization. By harnessing the power of retrieval-based models, RAG can provide more precise and comprehensive answers to complex questions, surpassing the constraints of traditional language models that generate responses based solely on a fixed-length context window. Similarly, for summarization tasks, RAG can create succinct summaries that encapsulate the core of the original text, while also incorporating pertinent information from external sources.

The contribution of LLMs and document retrieval in RAG is crucial. LLMs, such as OllamaLLM, lay the groundwork for generating human-like text, while the document retrieval component ensures that the generated text is not just plausible, but also factually accurate and contextually relevant.

In the upcoming sections, we will delve into how to construct a RAG-powered API using FastAPI and OllamaLLM, bringing the benefits of retrieval-augmented generation to your applications.

## 1. Setting Up the Environment

Instead of setting up a traditional Python environment, we will be using Docker to build our application. Docker allows us to package our application with all of its dependencies into a standardized unit for software development.

Our Docker setup will consist of two main services:

1. **Ollama**: This service will use the latest Ollama image and will be responsible for interacting with the Large Language Model (LLM), which in our case is llma3.2. The Ollama home directory will be mounted as a volume for persisting and reusing data.

2. **FastAPI**: This service will build from the current directory and will be responsible for handling REST requests. It will depend on the Ollama service and expose port 9001 for communication. It will also mount two volumes: one for the index path and another for the local documents path.

The Vector Database (Chroma) will be used within the FastAPI container.

Here is an example of the Docker Compose file:

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-container
    volumes:
      - ${HOST_OLLAMA_HOME}:/root/.ollama
    restart: always

  fastapi:
    build: .
    container_name: fastapi-container
    restart: always
    depends_on:
      - ollama
    ports:
      - "9001:9001"
    volumes:
      - ${HOST_INDEX_PATH}:${INDEX_PATH}
      - ${HOST_DOCS_PATH}:${LOCAL_DOCS_PATH}
```

We will also use an `.env` file to manage our environment variables. These variables will be used to configure various aspects of our application, such as the index path, the LLM model, the embedding model name, the splitter chunk size and overlap, and the paths for the local documents. Here's an example of what the `.env` file might look like:

```bash
INDEX_PATH=/app/index/
HOST_INDEX_PATH=./index/

LLM_MODEL=llama3.2:1b
LLM_HOST=ollama
LLM_PORT=11434
HOST_OLLAMA_HOME=/path/to/your/ollama/home/

EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
SPLITTER_CHUNK_SIZE=1000
SPLITTER_CHUNK_OVERLAP=200

HOST_DOCS_PATH=/path/to/your/local/docs/
LOCAL_DOCS_PATH=/docs/
```

## 2. Creating the API Structure

In this section, we will explain the basic structure of the FastAPI application. We will start by defining the `RequestModel` for our API. This model will define the query parameters that our API will accept. Here's what it looks like:

```python
class RequestModel(BaseModel):
    #prompt is mandatory
    prompt: str
    # prompt_template is optional
    prompt_template: Optional[str] = Field(None)
    session_context: Optional[str] = Field(None)
```

Next, we will introduce the `create_index` function. This function will interact with the vector database. It reads all the PDF files in the `local_docs` folder, uses HuggingFaceEmbeddings to create embeddings, and stores them in the Chroma vector database. The database will persist the data in the folder defined by `INDEX_PATH`.

Embeddings are a type of word representation that allows words with similar meaning to have a similar representation. They are a distributed representation for text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging natural language processing problems. Hugging Face's sentence transformer is a powerful tool that generates these embeddings. You can learn more about embeddings [here](https://en.wikipedia.org/wiki/Word_embedding) and about Hugging Face's sentence transformer [here](https://www.sbert.net/).

## 3. Implementing the /query API

The `/query` API is the main endpoint of our application that implements the RAG. It interacts with the LLM using Ollama APIs. Here's a simplified explanation of the code:

```python
@app.post("/query/")
async def generate_response(request: RequestModel):
    ...
```

This line defines the `/query` endpoint for our API. It's a POST endpoint that accepts a `RequestModel` as input.

```python
    session_context = ""
    if request.session_context:
        session_context = request.session_context
    if len(session_context) > 0:
        local_prompt = session_context + "\n" +request.prompt
    else:
        local_prompt = request.prompt
```

This block of code handles the session context. If a session context is provided in the request, it is appended to the prompt.

```python
    try:
        prompts = []
        prompts.append(local_prompt)
        #if retriever is None then generate the response without it
        if vector_db is None:
            return llm.generate(prompts)
        else:
            retriever = vector_db.as_retriever()
            docs = retriever.invoke(local_prompt)
            # Check if there are hits in the vector_db
            if not docs:
                return llm.generate(prompts)
```

This block of code generates the response. If a vector database is available, it retrieves relevant documents and uses them to generate the response. If no relevant documents are found, it generates the response without them.

```python
            prompt_template = rag_prompt_template
            if request.prompt_template:
                prompt_template = PromptTemplate.from_template(request.prompt_template+"""
                                                                Question: {question}
                                                                Context: {context}
                                                                Answer:""")
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            res = rag_chain.invoke(local_prompt)
            output = {"response":res,
                    "session_context":session_context}
            return output
    except Exception as e :
        print(str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
```

This block of code handles the RAG chain. It sets up the prompt template, creates the RAG chain, invokes it with the local prompt, and returns the generated response.

In the next section, we'll start building our RAG-powered API.
```
