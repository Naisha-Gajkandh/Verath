import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch


# ── Override settings before the app imports them ────────────────────────────
import os
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-for-validation")
os.environ.setdefault("DATABASE_NAME", "Verath_test")


from app.main import app
from app.services.auth import create_access_token, create_refresh_token


# ── Event loop ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── HTTP client ───────────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def client():
    """Async test client wired directly to the FastAPI app — no real server needed."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


# ── Auth helpers ──────────────────────────────────────────────────────────────
@pytest.fixture
def test_username():
    return "test_user"


@pytest.fixture
def access_token(test_username):
    return create_access_token(test_username)


@pytest.fixture
def refresh_token(test_username):
    return create_refresh_token(test_username)


@pytest.fixture
def auth_headers(access_token):
    return {"Authorization": f"Bearer {access_token}"}


# ── MongoDB mock ──────────────────────────────────────────────────────────────
@pytest.fixture
def mock_db(monkeypatch):
    """
    Patches all MongoDB collections used in services so tests never
    touch a real database.
    """
    mock_col = MagicMock()
    mock_col.find_one = AsyncMock(return_value=None)
    mock_col.insert_one = AsyncMock(return_value=MagicMock(inserted_id="mock_id"))
    mock_col.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    mock_col.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    mock_col.find = MagicMock(return_value=_async_cursor([]))
    mock_col.aggregate = MagicMock(return_value=_async_cursor([]))
    return mock_col


class _async_cursor:
    """Minimal async cursor that yields from a list."""
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration

    def sort(self, *args, **kwargs):
        return self


# ── ChromaDB mock ─────────────────────────────────────────────────────────────
@pytest.fixture
def mock_chroma(monkeypatch):
    """Prevents ChromaDB from touching disk during tests."""
    mock_col = MagicMock()
    mock_col.upsert = MagicMock()
    mock_col.query = MagicMock(return_value={
        "ids": [["mem_1", "mem_2"]],
        "documents": [["Memory about a meeting", "Memory about a deadline"]],
        "metadatas": [[
            {"intent": "meeting", "speaker": "unknown", "importance": 0.8,
             "lifecycle": "short_term", "timestamp": "2024-01-15T15:00:00", "user_id": "test_user"},
            {"intent": "deadline", "speaker": "unknown", "importance": 0.9,
             "lifecycle": "short_term", "timestamp": "2024-01-16T09:00:00", "user_id": "test_user"},
        ]],
        "distances": [[0.1, 0.2]],
    })
    mock_col.count = MagicMock(return_value=2)
    mock_col.delete = MagicMock()
    mock_col.get = MagicMock(return_value={"ids": [], "metadatas": [], "documents": [], "embeddings": []})

    monkeypatch.setattr(
        "app.services.memory_store._get_collection",
        lambda user_id: mock_col
    )
    return mock_col


# ── Embedding mock ────────────────────────────────────────────────────────────
@pytest.fixture
def mock_embedding(monkeypatch):
    """Returns a deterministic fake embedding so tests don't call Gemini."""
    monkeypatch.setattr(
        "app.services.gemini_embedding.get_embedding",
        lambda text: [0.1] * 768
    )
    monkeypatch.setattr(
        "app.services.memory_store.get_embedding",
        lambda text: [0.1] * 768
    )
