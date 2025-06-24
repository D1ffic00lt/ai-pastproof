<h1 align="center">PastProof AI — ML Module of the Automated Fact‑Checking System</h1>
<div align="center">
	<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/D1ffic00lt/ai-pastproof">
	<img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/D1ffic00lt/ai-pastproof">
	<img alt="GitHub commits stats" src="https://img.shields.io/github/commit-activity/y/D1ffic00lt/ai-pastproof">
</div>
<p align="center">
<strong>PastProof</strong> AI is the machine‑learning core of the PastProof fact‑checking service.
</p>
<p align="center">
The module takes <strong>raw text</strong> as input and returns <strong>only those claims it determines to be 
false</strong>, 
each accompanied by the evidence passage that refutes it and an optional natural‑language explanation.
</p>

---

## Pipeline at a Glance

| Step                            | Purpose                                                                      | Key dependencies                 |
| ------------------------------- | ---------------------------------------------------------------------------- | -------------------------------- |
| 1. Pre‑processing               | Cleaning, normalisation, sentence splitting, optional coreference resolution | `spaCy`                          |
| 2. Semantic Retrieval           | k‑NN search for candidate evidence paragraphs (FAISS + SentenceTransformers) | `faiss`, `sentence-transformers` |
| 3. Heuristic Filtering          | Rule‑based filters (dates, NER, length) to prune irrelevant matches          | `spaCy`, `regex`                 |
| 4. Cross‑Encoder Classification | DeBERTa / RoBERTa cross‑encoder scores each (claim, evidence) pair           | `sentence-transformers`          |
| 5. Explanation LLM (optional)   | Generates a concise rationale for why the claim is false                     | `transformers`                   |
| 6. Aggregation                  | Collects only the `REFUTED` items and emits them via `SuggestionResponse`    | —                                |

`ai_services/response.py` defines the `SuggestionResponse` dataclass that standardises the API output, and
`ai_services/sentence.py` contains utilities for robust sentence segmentation that are reused across the pipeline.

---

## Repository Layout

```
ai-pastproof/
├─ ai_services/
│   ├─ __init__.py
│   ├─ interfaces.py
│   ├─ response.py          # API‑level response schema
│   ├─ sentence.py          # sentence‑splitting helpers
│   ├─ preprocessing.py     # text pre‑processing pipeline
│   ├─ vector_storage.py    # FAISS wrapper
│   ├─ utils.py             # misc helpers (e.g. progress‑bar suppression)
│   ├─ typing.py            # shared type aliases
│   ├─ static.py            # prompt templates
│   └─ models/
│       ├─ __init__.py
│       ├─ coref.py         # coreference resolver wrapper
│       ├─ fact_checker.py  # main FactCheckerPipeline
│       └─ explanation.py   # LLM explanation helper
│
├─ pyproject.toml
├─ .gitmodules
└─ README.md
```

---

## Quick Start

### Minimal Example (Config‑Driven)

```python
# config.py — consolidated settings
DEVICE = "cuda"
PROCESSING_DEVICE = "cuda"
STORAGE_SEARCH_K = 7
STORAGE_SEARCH_THRESHOLD = 0.7
AUTOMATIC_CONTEXTUALIZATION = True
NER_CORPUS = "en_core_web_trf"
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
ENABLE_NER = True
SENTENCE_TRANSFORMER_MODEL = "intfloat/e5-base-v2"
STORAGE_PATH = "data/corpus.index"
SPACY_CORE = "en_core_web_trf"
ENABLE_LLM = False
```

```python
import spacy
from sentence_transformers import SentenceTransformer

from ai_services.vector_storage import VectorStorage
from ai_services.models.fact_checker import FactCheckerPipeline
from ai_services.preprocessing import Pipeline
from ai_services.models.coref import CorefResolver
import config  # the file above

def make_fact_checker() -> FactCheckerPipeline:
    spacy.prefer_gpu()
    encoder = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL,
                                  device=config.DEVICE)

    storage = VectorStorage(
        dim=encoder.get_sentence_embedding_dimension(),
        embedder=encoder.encode
    )
    storage.load(config.STORAGE_PATH)

    coref_pipeline = Pipeline(
        coref=CorefResolver(sentence_splitter=config.SPACY_CORE,
                            device=config.DEVICE),
        device=config.DEVICE
    )

    return FactCheckerPipeline(
        vector_storage=storage,
        processing_pipeline=coref_pipeline,
        processing_device=config.PROCESSING_DEVICE,
        device=config.DEVICE,
        get_explanation=config.ENABLE_LLM,
        storage_search_k=config.STORAGE_SEARCH_K,
        storage_search_threshold=config.STORAGE_SEARCH_THRESHOLD,
        automatic_contextualisation=config.AUTOMATIC_CONTEXTUALIZATION,
        ner_corpus=config.NER_CORPUS,
        model_name=config.MODEL_NAME,
        enable_ner=config.ENABLE_NER
    )

checker = make_fact_checker()

text = """\
Napoleon was crowned emperor in 1815.
The Battle of Hastings happened in 1066.
"""

false_claims = checker(text)          # returns only the refuted statements
print(false_claims.json(indent=2))    # SuggestionResponse (see below)
```

Only the incorrect claim is returned, correct statements are silently ignored.

---

## Limitations

* Accuracy depends on the completeness and quality of the corpus.
* Pre‑trained models cover English and partly Russian, other languages require retraining.
* LLM explanations may hallucinate, always inspect the evidence links.
