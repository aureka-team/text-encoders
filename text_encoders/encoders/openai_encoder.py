import numpy as np

from typing import Optional
from openai import OpenAI, BadRequestError

from more_itertools import flatten
from tiktoken import encoding_for_model

from common.logger import get_logger
from text_encoders.meta import TextEncoder
from text_encoders.weaviate_cache import WeaviateCache


logger = get_logger(__name__)


class OpenAIEncoder(TextEncoder):
    def __init__(
        self,
        batch_size: int = 256,
        max_concurrency: int = 5,
        model_name: str = "text-embedding-3-large",
        dimensions: int = 1024,
        timeout: int = 10,
        weaviate_cache: Optional[WeaviateCache] = None,
        tokenizer_model: str = "gpt-4o",
    ):
        super().__init__(
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            weaviate_cache=weaviate_cache,
        )

        self.openai_client = OpenAI(timeout=timeout)
        self.model_name = model_name
        self.dimensions = dimensions
        self.tokenizer = encoding_for_model(tokenizer_model)

    def _get_num_tokens(self, texts: list[str]) -> int:
        tokens = self.tokenizer.encode_batch(texts)
        return len(list(flatten(tokens)))

    def _encode(self, texts: list[str]) -> np.ndarray:
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=self.model_name,
                dimensions=self.dimensions,
            )

        except BadRequestError as error:
            texts_n_tokens = (
                {
                    "text": text,
                    "n_tokens": self._get_num_tokens(texts=[text]),
                }
                for text in texts
            )

            sorted_texts_n_tokens = sorted(
                texts_n_tokens,
                key=lambda x: x["n_tokens"],
                reverse=True,
            )

            logger.error(
                f"error encoding texts, longest text => {sorted_texts_n_tokens[0]}"
            )

            raise error

        embeddings = [data_item.embedding for data_item in response.data]
        return np.array(embeddings)
