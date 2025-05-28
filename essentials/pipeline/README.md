# LangChain-GPT Pipeline

## 📌 Project Overview
This pipeline is designed to process scientific PDFs by extracting text, classifying sections, scoring them based on relevance, summarizing key sections, and fetching related papers. It is part of the LangChain-GPT project, aimed at enhancing literature search and analysis capabilities.

## 📂 Directory Structure
```
/pipeline
│
├── pipeline_runner.py          # Main script to run the entire pipeline
├── pdf_to_document.py         # Parses PDF and extracts text and metadata
├── section_classifier.py      # Classifies sections using a trained ML model
├── ranking_engine.py          # Scores sections based on relevance and other factors
├── summarizer_wrapper.py      # Summarizes key sections
└── semantic_search_connector.py # Fetches related papers using Semantic Scholar API
```

## 🚀 Pipeline Workflow
1. **PDF Input**: The pipeline starts by taking a PDF file as input.
2. **Text Extraction**: `pdf_to_document.py` extracts text and metadata from the PDF.
3. **Section Classification**: `section_classifier.py` classifies sections into categories like Introduction, Methods, etc.
4. **Scoring**: `ranking_engine.py` scores sections based on relevance, recency, and citation count.
5. **Summarization**: `summarizer_wrapper.py` summarizes the top sections.
6. **Document Enrichment**: `semantic_search_connector.py` fetches related papers to enrich the document.

## 🤖 Section Classifier Details
The section classifier was trained in Phase 2.1 using features like TF-IDF and models such as SVM/Logistic Regression. It predicts labels like Introduction, Methods, Results, Discussion, and Conclusion. The trained model is loaded in this pipeline for inference.

## 🛠️ How to Run
To run the pipeline, use the following command:
```bash
python pipeline_runner.py <path_to_pdf>
```

## 📌 Dependencies
- PyMuPDF
- Transformers
- Torch
- Requests
- Scikit-learn

## 📌 Future Enhancements
This pipeline will integrate with later phases of the LangChain-GPT project, including RAG systems, LLMs, and entity extraction.

## 👨‍💻 Author Credit
This pipeline is part of the LangChain-GPT project, developed to enhance scientific literature processing and analysis. 