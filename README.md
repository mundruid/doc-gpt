# doc-gpt

[deprecated] This code needs to be update to laetst packages.
Code used from the blog [Run GPT-style questions over your own files](https://www.reaminated.com/run-chatgpt-style-questions-over-your-own-files-using-the-openai-api-and-langchain) as a base. This code will read your personal files and answer questions based on their content.

# pdf-gpt

Reads pdf files from one directory, transforms them to embeddings and loads to Chroma DB. 

Code modified from blog [Implementing RAG in Langchain and Croma DB](https://medium.com/@callumjmac/implementing-rag-in-langchain-with-chroma-a-step-by-step-guide-16fc21815339).

## TODO

- [X] Extend to load a whole directory
- [ ] Load files from multiple directories
- [X] Extend for pdf files
- [ ] Extend for word files
- [ ] Use a local model
- [ ] Semantic chunking
- [ ] Question chaining to use the context of previous questions