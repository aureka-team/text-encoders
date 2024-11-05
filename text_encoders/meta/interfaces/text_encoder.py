import asyncio

import numpy as np

from tqdm import tqdm
from typing import Optional

from abc import ABC, abstractmethod
from more_itertools import chunked, flatten

from common.logger import get_logger
from text_encoders.weaviate_cache import WeaviateCache


logger = get_logger(__name__)


class TextEncoder(ABC):
    def __init__(
        self,
        batch_size: int = 1024,
        max_concurrency: int = 10,
        weaviate_cache: Optional[WeaviateCache] = None,
    ):
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.weaviate_cache = weaviate_cache

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _get_n_tokens(self, texts: list[str]) -> int:
        pass

    @abstractmethod
    def _encode(self, texts: list[str]) -> np.ndarray:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.weaviate_cache is None:
            return self._encode(texts=texts)

        loaded_vectors = self.weaviate_cache.load(texts=texts)
        uncached_indexes = {
            idx for idx, vector in enumerate(loaded_vectors) if vector is None
        }

        if not uncached_indexes:
            return np.array(loaded_vectors)

        uncached_texts = [
            text for idx, text in enumerate(texts) if idx in uncached_indexes
        ]

        uncached_vectors = self._encode(texts=uncached_texts)
        self.weaviate_cache.save(
            texts=uncached_texts,
            vectors=uncached_vectors,
        )

        for idx, uncached_vector in zip(uncached_indexes, uncached_vectors):
            loaded_vectors[idx] = uncached_vector

        assert len(texts) == len(loaded_vectors)
        return np.array(loaded_vectors)

    def batch_encode(self, texts: list[str]) -> np.ndarray:
        n_tokens = self._get_n_tokens(texts=texts)
        if len(texts) <= self.batch_size:
            return self.encode(texts=texts)

        text_chunks = chunked(texts, self.batch_size)
        chunk_vectors = map(
            self.encode,
            tqdm(
                text_chunks,
                total=(len(texts) // self.batch_size),
                desc=f"encoding {n_tokens} tokens",
                ascii=" ##",
                colour="#808080",
            ),
        )

        text_vetors = np.array(list(flatten(chunk_vectors)))
        return text_vetors

    async def async_encode(
        self,
        texts: list[str],
        pbar: tqdm | None = None,
    ) -> np.ndarray:
        async with self.semaphore:
            vectors = await asyncio.to_thread(self.encode, texts)
            if pbar is not None:
                pbar.update(1)

            return vectors

    async def async_batch_encode(self, texts: list[str]) -> np.ndarray:
        n_tokens = self._get_n_tokens(texts=texts)
        if len(texts) <= self.batch_size:
            return self.encode(texts=texts)

        text_chunks = chunked(texts, self.batch_size)
        with tqdm(
            text_chunks,
            total=(len(texts) // self.batch_size),
            desc=f"encoding {n_tokens} tokens",
            ascii=" ##",
            colour="#808080",
        ) as pbar:
            async_tasks = [
                self.async_encode(texts, pbar=pbar) for texts in text_chunks
            ]

            chunk_vectors = await asyncio.gather(*async_tasks)

        text_vetors = np.array(list(flatten(chunk_vectors)))
        return text_vetors
