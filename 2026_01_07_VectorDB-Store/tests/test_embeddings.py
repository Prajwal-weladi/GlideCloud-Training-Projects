import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import anyio
from app.services import embeddings


def test_generate_embedding():
    async def run_test():
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [1.0, 2.0, 3.0]}
        mock_response.raise_for_status.return_value = None

        with patch(
            "httpx.AsyncClient.post",
            new=AsyncMock(return_value=mock_response)
        ):
            result = await embeddings.generate_embedding("hello")
            assert result == [1.0, 2.0, 3.0]

    anyio.run(run_test)
