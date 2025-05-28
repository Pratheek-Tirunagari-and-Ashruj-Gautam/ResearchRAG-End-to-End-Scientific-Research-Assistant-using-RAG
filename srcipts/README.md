# LangChainGPT - Autonomous Research Agent

An autonomous research agent for scientific papers built using LangChain.

## Project Overview

LangChainGPT is a research agent designed to process, analyze, and extract insights from scientific papers. This project is being developed in phases, with Phase 1 focusing on setting up the infrastructure without making external API calls.

### Features (Planned)

- Document processing and text extraction from PDFs
- Text cleaning and preprocessing
- Integration with LLM APIs (OpenAI, HuggingFace, etc.)
- Research question analysis and answering
- Citation management

## Project Structure

```
langchain-gpt/
├── config/                 # Configuration files
├── data/                   # Sample data storage
├── docker/                 # Docker configurations
│   ├── docker-compose.yml  # Docker compose for services
│   └── Dockerfile.dev      # Development container
├── notebooks/              # Jupyter notebooks for experiments
├── src/                    # Source code
│   └── langchain_gpt/      # Main package
│       ├── api_clients/    # API client interfaces
│       ├── config/         # Configuration modules
│       ├── document_processing/ # Document processing modules
│       └── utils/          # Utility functions
├── tests/                  # Test directory
│   └── unit/               # Unit tests
├── .env.template           # Template for environment variables
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
└── README.md               # Project documentation
```

## Installation

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for running services)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/langchain-gpt.git
   cd langchain-gpt
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Run the setup script to create directories and environment file:
   ```bash
   python setup_project.py
   ```
   
   This script will:
   - Create all required directories
   - Create a `.env` file from the template if it doesn't exist
   - Verify that required packages are installed

5. Start services using Docker (optional for development):
   ```bash
   cd docker
   docker-compose up -d
   ```
   
6. Verify the project structure:
   ```bash
   python verify_structure.py
   ```

## Usage

### Document Processing

Process a PDF document:

```python
from langchain_gpt.document_processing.document_processor import DocumentProcessor

processor = DocumentProcessor()
document = processor.process_file("path/to/document.pdf")

# Access document chunks
for chunk in document.chunks:
    print(chunk.text)
```

## Development

### Running Tests

```bash
pytest tests/
```

### Docker Development Environment

Build and run the development container:

```bash
cd docker
docker build -t langchaingpt-dev -f Dockerfile.dev ..
docker run -it -v $(pwd)/..:/app langchaingpt-dev
```

## Phase 1 Roadmap

- [x] Project structure setup
- [x] Python virtual environment configuration
- [x] Core infrastructure implementation
- [x] Document processing module
- [x] API client interfaces
- [x] Docker setup
- [x] Directory creation utilities
- [x] Project setup and verification scripts
- [ ] Unit tests for all modules
- [ ] Example Jupyter notebooks

## Phase 2 Goals

- Implement API clients for real services
- Add document parsing with Grobid
- Develop embedding storage
- Create basic research agent

## Contributing

Contributions are welcome! This is a student project, so please be kind and constructive.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 