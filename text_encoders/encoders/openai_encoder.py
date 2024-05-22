import numpy as np

from openai import OpenAI
from typing import Optional

from common.logger import get_logger
from text_encoders.meta import TextEncoder


logger = get_logger(__name__)


class OpenAIEncoder(TextEncoder):
    def __init__(
        self,
        batch_size: int = 256,
        max_concurrency: int = 5,
        model_name: str = "text-embedding-3-large",
        dimensions: int = 1024,
        cache_path: Optional[str] = "/resources/cache/embeddings",
    ):
        super().__init__(
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            cache_path=cache_path,
        )

        self.openai_client = OpenAI()
        self.model_name = model_name
        self.dimensions = dimensions

    def _encode(self, texts: list[str]) -> np.ndarray:
        response = self.openai_client.embeddings.create(
            input=texts,
            model=self.model_name,
            dimensions=self.dimensions,
        )

        embeddings = [data_item.embedding for data_item in response.data]
        embeddings = np.array(embeddings)

        return embeddings
