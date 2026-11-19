import numpy as np
import pytest

from config import CONFIG
from embeddings.embedding_model import get_embedding_model


@pytest.fixture(scope="module")
def model():
    return get_embedding_model()


def test_dimension_and_normalisation(model):
    v = model.encode_documents(["graph neural networks", "image segmentation with transformers"])
    assert v.shape == (2, model.dim) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_semantic_ordering(model):
    q = model.encode_queries(["traffic forecasting with graph neural networks"])[0]
    docs = model.encode_documents(["We forecast road traffic using spatio-temporal graph convolutions.",
                                   "We study the syntax of Arabic dialects in social media."])
    sims = docs @ q
    assert sims[0] > sims[1]


def test_max_seq_length_fits_chunks(model):
    assert model.model.max_seq_length >= CONFIG["chunk_size"]
