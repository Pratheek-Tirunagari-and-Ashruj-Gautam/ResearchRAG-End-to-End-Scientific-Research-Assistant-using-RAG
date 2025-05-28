# Phase 2.2: Scientific NLP Pipeline

This module implements a comprehensive Scientific NLP Pipeline with three main components:

1. **Entity Extraction** - Extract scientific entities from research papers
2. **Relation Extraction** - Identify relationships between entities using dependency parsing
3. **Claim Detection** - Find and evaluate scientific claims in results/discussion sections

## Components

### Entity Extraction (`entity_extraction.py`)

Uses SciSpacy models to identify domain-specific scientific entities from text.

- Default model: `en_core_sci_sm`
- Entities are extracted with their text, type, position, and sentence context
- Each document is processed and augmented with entity information

### Relation Extraction (`relation_extraction.py`)

Implements a pattern-based relation extractor using SpaCy's dependency parsing.

- Identifies key relations like "uses [method]", "measured [quantity]", etc.
- Each relation has a type, verb, arguments, and confidence score
- Extracts subject-verb-object patterns for scientific assertions

### Claim Detection (`claim_detection.py`)

Detects scientific claims using linguistic patterns and heuristics.

- Identifies claim verbs, hedging language, confidence boosters, and evidence
- Assigns confidence scores to each identified claim
- Prioritizes claims in results, discussion, and conclusion sections

### Pipeline Integration (`run_phase2_2_pipeline.py`)

Combines all three components into a unified pipeline.

- Processes documents in sequence: entities → relations → claims
- Handles batched processing for large datasets
- Provides detailed statistics and logs of the extraction process

## Usage

1. Run the pipeline with default settings (processes all documents):

```bash
python run_phase2_2_pipeline.py
```

2. Run with specific parameters:

```bash
python run_phase2_2_pipeline.py --input [INPUT_FILE] --output-dir [OUTPUT_DIR] --max-docs [NUM_DOCS]
```

3. Or use the batch file for interactive execution:

```bash
run_pipeline.bat
```

4. Run automated tests to verify all components are working correctly:

```bash
python test_phase2_2_pipeline.py
```

## Input/Output

- **Input**: JSON files from `data/transfer_learning/prepared/section_classification/`
- **Output**: Enhanced JSON files with entities, relations, and claims in `data/derived/phase2.2_output/`

## Output Format

Each processed document contains:

```json
{
  "text": "Original text...",
  "section_type": "...",
  "entities": [
    {
      "text": "entity text",
      "label": "entity type",
      "start": 0,
      "end": 10,
      "sent_idx": 0
    }
  ],
  "entity_count": 5,
  "relations": [
    {
      "type": "uses_method",
      "verb": "use",
      "arguments": [
        {"role": "subject", "text": "We"},
        {"role": "object", "text": "machine learning"}
      ],
      "confidence": 0.7
    }
  ],