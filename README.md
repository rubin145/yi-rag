## yi-rag

RAG para textos en yiddish con pipeline de preprocesado (split + traducción opcional), embeddings multi-provider y indexado en Qdrant con named vectors.

### Componentes
- `core/preprocess.py`: lee `data/raw/*.txt`, detecta páginas (`--- Page N ---`), extrae título (`Title:`), hace chunking por página/tokens/caracteres y traduce opcionalmente a EN. Guarda `data/processed/chunks.parquet` o partes.
- `core/embeddings.py`: cálculos de embeddings (OpenAI, Cohere, Gemini, opcional local ST/LaBSE).
- `scripts/index_corpus.py`: crea colección y sube puntos a Qdrant con named vectors desde `config/vectors.yaml`.

### Requisitos
- Python 3.12 (Poetry recomendado)
- Qdrant (Cloud o self-hosted) si vas a indexar.

### Configuración
Copia `.env.example` a `.env` y ajusta:
- Chunking: `CHUNK_STRATEGY`, `CHUNK_TOKENS`, `CHUNK_OVERLAP`, …
- Traducción: `TR_PROVIDER` (`none|openai|cohere|gemini`), claves y modelos.
- Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.

### Instalar
Con Poetry:

```bash
poetry install
```

### Preprocesar
Genera el parquet (sin traducir por defecto):

```bash
poetry run python -m core.preprocess
```

Modos de guardado (env `PREPROCESS_INCREMENTAL`): `none` (default), `book` (un parquet por libro), `append` (apéndice a un parquet global).

### Indexar a Qdrant
Define tus vectores en `config/vectors.yaml` y ejecuta:

```bash
poetry run python scripts/index_corpus.py --parquet data/processed/chunks.parquet --vectors-yaml config/vectors.yaml --recreate
```

Usa `--only <vector_name>` para subir un vector específico y `--parts data/processed/parts` si usaste modo por libro.

### Datos de ejemplo
- Coloca `.txt` en `data/raw/`. Formato opcional: separadores `--- Page N ---` y encabezado `Title: ...`.
- Metadata opcional en `data/metadata.csv` con columnas: `book_id,title_yi,title_en,author,year`.

### Notas
- Gemini embeddings usan el SDK nuevo `google-genai` para batch; la traducción con Gemini usa `google-generativeai`.
- Campos principales del parquet: `yi_text`, `tr_en_text`, `title_yi`, `title_en`, `book_id`, `page`, `chunk_idx`.

