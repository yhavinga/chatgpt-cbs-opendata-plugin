from typing import Dict, List

import numpy as np
import pytest


def mock_get_embeddings(texts: List[str]) -> List[np.ndarray]:
    def hash_text(text: str) -> float:
        # hash the text to a number between -0.9 and 0.9
        total = sum(ord(char) for char in text)
        return -0.9 + (total % 18) * 0.1

    num = 1536
    step = 100 * np.pi / num
    angles = np.arange(0, 100 * np.pi, step)
    sin_values = 0.9 * np.sin(angles)
    return [np.round(hash_text(text) + sin_values, 1).tolist() for text in texts]


def mock_get_chat_completion(messages: List[Dict]) -> str:
    if "column1 is greater than 50" in messages[-1]["content"]:
        return "result = df[df['column1'] > 50]"
    return ""


@pytest.fixture()
def mock_openai(monkeypatch):
    # NB: we need to mock the get_embeddings that's imported in the chunk's module,
    # not the openai module
    monkeypatch.setattr("server.table.get_embeddings", mock_get_embeddings)
    monkeypatch.setattr("server.table.get_chat_completion", mock_get_chat_completion)
