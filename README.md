# Text Encoders

This library provides tools to work with text encoders, such as OpenAI API embeddings and open-source embeddings (support for open-source embeddings is pending). It includes functionality for batch encoding, synchronous and asynchronous requests to OpenAI, and embedding caching using Weaviate.

---

## Features

-   Support for OpenAI API embeddings.
-   Batch encoding for efficient processing of large text datasets.
-   Asynchronous batch encoding with configurable concurrency.
-   Embedding caching via Weaviate for faster retrieval and reduced API costs by avoiding redundant encoding requests.

---

## Setup example with uv

1.  Install `uv` by following the [installation guide.](https://docs.astral.sh/uv/getting-started/installation/)
2.  Install `text-encoders` using uv:

    ```bash
    uv python install 3.12
    uv venv
    uv pip install git+https://git@github.com/aureka-team/text-encoders.git

    ```

---

## Usage

### Importing Modules

```python
from text_encoders.encoders import OpenAIEncoder
from text_encoders.weaviate_cache import WeaviateCache
```

### Setting Up the Encoder

```python
weaviate_cache = WeaviateCache()  # Initialize the embedding cache.
openai_encoder = OpenAIEncoder(
    batch_size=1024,  # Number of texts processed in each batch.
    max_concurrency=10,  # Maximum number of concurrent asynchronous requests.
    weaviate_cache=weaviate_cache  # Optional: Enable caching to reuse embeddings.
)
```

### Encoding Examples

```python
texts = [
    "This is the last example.",
    "And this is another example!"
]
```

Encodes all texts in a single request:

```python
embeddings = openai_encoder.encode(texts)
```

Encodes all texts in batches of `batch_size` sequentially:

```python
embeddings = openai_encoder.batch_encode(texts)
```

Encodes all texts in batches of `batch_size` asynchronously with a maximum concurrency of `max_concurrency`:

```python
embeddings = await openai_encoder.async_batch_encode(texts)
```

---

# Future Enhancements

-   Support for open-source embeddings.
-   Additional caching backends.
-   Extended encoding options for broader use cases.

---

# Contributing

Contributions are welcome! Please submit a pull request or open an issue if you encounter any bugs or have suggestions for new features.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
