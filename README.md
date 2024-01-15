# QA-RAG
Code for the paper, From RAG to QA-RAG: Integrating Generative AI for Pharmaceutical Regulatory Compliance Process

This repository contains the source code for the QA-RAG (Question and Answer Retrieval Augmented Generation) model, a tool designed for the pharmaceutical regulatory compliance domain. It integrates generative AI and RAG methodologies to efficiently navigate complex regulatory guidelines, providing accurate and relevant information through a Q&A format.

---

### Getting Started

1. **Install Required Packages**
   
    Install all the necessary Python packages listed in `requirements.txt`.
  
    ```python
    !pip install -r requirements.txt
    ```

2. **Initialize Models**
    
   Run the script `initialize_model.py` to download and set up the embedding and reranker models.
      
    ```python
    !python initialize_model.py
    ```

3. **Run the Model**
   
   Use `main.py` to execute the model. Here's the basic command structure:
      
    ```python
    !python main.py --question "<Your Question>" --top_k <Top K Docs> --top_k_ans <Top K Docs from Answer> --num_docs <Number of Initial Docs> --num_docs_ans <Number of Initial Docs from Answer> --final_k <Final Number of Docs> --method <Specific method to use>
    ```

    --question: Your query related to the guidelines.
   
    --top_k: Number of top documents to retrieve initially.
   
    --top_k_ans: Number of top documents to retrieve based on the model-generated answer.
   
    --num_docs: Total number of documents to retrieve initially.
   
    --num_docs_ans: Total number of documents to retrieve based on the answer.
   
    --final_k: Number of final top documents to return after reranking.
   
    --method: Specific method to use. Choose the method to use: QA_RAG, Multiquery, or HyDE
   

    **Example**
   
    ```python
    !python main.py --question "How can I get the information of FDA?" --top_k 3 --top_k_ans 3 --num_docs 5 --num_docs_ans 5 --final_k 2 --method QA_RAG
    ```

---

### Repository Structure
**main.py**: The main script to run the QA-RAG model.

**config.py**: Configuration file containing essential settings.

**embeddings.py**: Script to handle embedding processes.

**model.py**: Contains the main logic for the Document Retriever, Answer Generator, and other components.

**reranker.py**: Handles the reranking of retrieved documents.

**initialize_model.py**: Initializes and sets up models.

---

### Experimental Data
**Dataset**: The dataset that was used in the experiments in the paper is included. It's composed of real-world questions and answers related to FDA guidelines.

**Results Data**: Alongside the dataset, the results data obtained from running the experiments is included
