# 🌍 ResearchRAG: End-to-End Scientific Research Assistant using RAG & LLMs

An advanced Retrieval-Augmented Generation (RAG) pipeline for scientific document understanding and question answering. Built with modular components for PDF parsing, entity extraction, citation analysis, semantic retrieval, and response generation via local LLMs.
## ✍️ Authors
### Ashruj Gautam

### Pratheek Tirunagari

> 📚 **Run the pipeline with just one file:** `app.py`  
> This triggers `pipeline_runner.py`, which internally coordinates all steps across modules.

---

## 🌐 Demo 


---

## ⚖️ Features
- Scientific PDF parsing with section/figure/reference extraction  
- Named entity, relation & claim detection using spaCy & SciSpacy  
- Section classification via fine-tuned BERT  
- Layout-aware figure detection using LayoutLM  
- Semantic query expansion with BERT  
- Citation graph construction and scoring  
- Embedding & hybrid retrieval with ChromaDB  
- Prompt construction and budgeted context building  
- Final LLM answer generation via local Mistral (GGUF format)

---

## 📄 Pipeline Overview (Step-by-Step)

### Step 1: PDF Parsing
- **Function:** `parse_pdf` (`pdf_to_document.py`)  
- Uses Grobid for metadata, tokenizes content, and prepares chunks

### Step 1.5: Scientific NLP Enrichment
- **Entity Extractor:** `ScientificEntityExtractor`  
- **Relation Extractor:** `PatternRelationExtractor`  
- **Claim Extractor:** `ScientificClaimDetector`  
- Adds entities, relations, and claims to section metadata

### Step 2: Section Classification & Chunking
- **Function:** `classify_sections`  
- Label sections (Introduction, Methods...)  
- Chunk and embed text for ChromaDB

### Step 2.5: Figure Detection
- **Functions:** `detect_figures_in_pdf`, `convert_figures_to_chunks`  
- Use LayoutLM to detect captions, convert to retrievable chunks

### Step 2.6: Reference Cleaning
- In `pipeline_runner.py`: normalizes raw references to title, authors, journal, etc.

### Step 3: Section Scoring
- **Function:** `score_sections`  
- Based on citation count, recency, and word count

### Step 4: Section Summarization
- **Function:** `summarize_sections`  
- One-line summaries per section

### Step 5: Fetch Related Papers
- **Function:** `fetch_additional_documents`  
- Stubbed integration with arXiv / Semantic Scholar

### Step 5.1: Query Expansion
- **Class:** `BERTQueryExpander`  
- BERT-based semantic keyword expansion

### Step 5.2: Citation Network Analysis
- **Class:** `CitationNetwork`  
- Builds citation graph and extracts influence clusters

### Step 5.3: Result Ranking
- **Class:** `FieldSpecificSearchOptimizer`  
- Ranks related papers for context enrichment

### Step 6: Embedding & Retrieval
- **Classes:** `ScientificEmbedding`, `Retriever`  
- Retrieve top-k relevant chunks from ChromaDB using a hybrid search

### Step 7: Context Building
- **Class:** `ContextBuilder`  
- Assembles context string for LLM with figure-aware logic

### Step 8: Prompt & LLM Generation
- **Functions:** `build_safe_prompt`, `safe_generate`, `LLMRunner`  
- Prepares the prompt and runs local LLM (Mistral) to get answers

### Step 9: Output
- **In:** `pipeline_runner.py`  
- Answer sent to `stdout` as: `LLM Answer: ...`

---

## 📊 Full Summary Table

| Step | Function/Class | Input | Output | Role |
|------|----------------|-------|--------|------|
| 1 | parse_pdf | PDF path | Document dict | PDF → structured text/metadata |
| 1.5 | Entity/Relation/Claim Extractors | Section text | Entities, relations, claims | NLP enrichment |
| 2 | classify_sections | Sections | Labeled + embedded sections | Section classification |
| 2.5 | detect_figures_in_pdf | PDF + tokenizer | Figure chunks | Figure caption detection |
| 2.6 | pipeline_runner.py | Raw references | Cleaned references | Metadata normalization |
| 3 | score_sections | Sections | Scored list | Rank importance |
| 4 | summarize_sections | Sections | One-line summaries | Context hinting |
| 5 | fetch_additional_documents | Paper title | Related papers | Context expansion |
| 5.1 | BERTQueryExpander | Keywords | Similar terms | Improve recall |
| 5.2 | CitationNetwork | Papers | Graph & communities | Citation analysis |
| 5.3 | FieldSpecificSearchOptimizer | Titles | Relevance scores | Re-ranking |
| 6 | Retriever + Embeddings | Query | Top-k chunks | Semantic retrieval |
| 7 | ContextBuilder | Chunks + Query | Context string | Prompt assembly |
| 8 | LLMRunner | Prompt + context | Final answer | LLM inference |
| 9 | pipeline_runner.py | LLM answer | stdout | UI output |

---

## 🎨 UI Demo 

- Launch the Streamlit app via `streamlit run app.py`  
- Paste a PDF, ask a question, see the result

---

## ♻️ Setup Instructions

```bash
# Clone the repo
$ git clone https://github.com/<your-org>/ResearchRAG-End-to-End-Scientific-Research-Assistant-using-RAG

# Navigate into the project
$ cd ResearchRAG-End-to-End-Scientific-Research-Assistant-using-RAG

# Create virtual environment
$ python -m venv LOLvenv && source LOLvenv/bin/activate

# Install dependencies
(LOLvenv) $ pip install -r requirements.txt

# Run the main app
(LOLvenv) $ python app.py
```
🚀 Tech Stack
Python 3.10+

LangChain-style modular architecture

ChromaDB for vector storage

Sentence Transformers (MiniLM)

LayoutLMv1 for figure detection

Fine-tuned BERT for section classification

Streamlit for interactive frontend

llama-cpp-python for local LLM execution
## 📦 Datasets Used

- **CORD-19**: Used for testing end-to-end scientific PDF parsing and question answering.
- **S2ORC (5K subset)**: Used to fine-tune our section classification model and evaluate semantic embedding quality.
- **PubLayNet**: Used for training our LayoutLM model for accurate figure detection in scientific PDFs.
- **arXiv Papers**: Sampled for testing document expansion, similarity ranking, and citation graph analysis.

## 📉 Performance
Task	Accuracy	F1
Section Classification	79.94%	
Reference Parsing	99.54%	

✨ Acknowledgments
Grobid: for metadata parsing

Hugging Face: for pretrained scientific models

ChromaDB: for open-source vector DB

Mistral: local LLM inference (GGUF)

