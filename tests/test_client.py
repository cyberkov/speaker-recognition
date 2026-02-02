"""Unit tests for speaker recognition client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from speaker_recognition.client import SpeakerRecognitionClient, SyncSpeakerRecognitionClient
from speaker_recognition.models import (
    AudioInput,
    HealthResponse,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
    VoiceSample,
)


class TestSpeakerRecognitionClientInit:
    """Tests for SpeakerRecognitionClient initialization."""

    def test_init_with_defaults(self) -> None:
        """Test client initialization with default timeout."""
        client = SpeakerRecognitionClient(base_url="http://localhost:8099")
        assert client._base_url == "http://localhost:8099"
        assert client._timeout == 30.0
        assert client._client is None

    def test_init_with_custom_timeout(self) -> None:
        """Test client initialization with custom timeout."""
        client = SpeakerRecognitionClient(base_url="http://localhost:8099", timeout=60.0)
        assert client._timeout == 60.0

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        client = SpeakerRecognitionClient(base_url="http://localhost:8099/")
        assert client._base_url == "http://localhost:8099"


class TestSpeakerRecognitionClientContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_creates_client(self) -> None:
        """Test that entering context manager creates client."""
        async with SpeakerRecognitionClient("http://localhost:8099") as client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_client(self) -> None:
        """Test that exiting context manager closes client."""
        client = SpeakerRecognitionClient("http://localhost:8099")
        async with client:
            internal_client = client._client
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_method(self) -> None:
        """Test explicit close method."""
        client = SpeakerRecognitionClient("http://localhost:8099")
        client._client = httpx.AsyncClient()
        await client.close()
        assert client._client is None


class TestSpeakerRecognitionClientHealthCheck:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SpeakerRecognitionClient("http://localhost:8099")
            result = await client.health_check()

            assert isinstance(result, HealthResponse)
            assert result.status == "healthy"
            mock_get.assert_called_once_with("/health")


class TestSpeakerRecognitionClientTrain:
    """Tests for training endpoint."""

    @pytest.mark.asyncio
    async def test_train_success(self) -> None:
        """Test successful training request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "trained_users": ["alice", "bob"],
            "count": 2,
        }

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            client = SpeakerRecognitionClient("http://localhost:8099")
            request = TrainingRequest(
                voice_samples=[
                    VoiceSample(
                        user="alice",
                        audio=AudioInput(audio_data="SGVsbG8=", sample_rate=16000),
                    ),
                    VoiceSample(
                        user="bob",
                        audio=AudioInput(audio_data="V29ybGQ=", sample_rate=16000),
                    ),
                ]
            )
            result = await client.train(request)

            assert isinstance(result, TrainingResult)
            assert result.status == "success"
            assert result.trained_users == ["alice", "bob"]
            assert result.count == 2
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_sends_correct_payload(self) -> None:
        """Test that training sends correct JSON payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "trained_users": ["alice"],
            "count": 1,
        }

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            client = SpeakerRecognitionClient("http://localhost:8099")
            request = TrainingRequest(
                voice_samples=[
                    VoiceSample(
                        user="alice",
                        audio=AudioInput(audio_data="dGVzdA==", sample_rate=22050),
                    )
                ]
            )
            await client.train(request)

            call_kwargs = mock_post.call_args
            assert call_kwargs[0][0] == "/train"
            json_payload = call_kwargs[1]["json"]
            assert "voice_samples" in json_payload
            assert json_payload["voice_samples"][0]["user"] == "alice"
            assert json_payload["voice_samples"][0]["audio"]["audio_data"] == "dGVzdA=="
            assert json_payload["voice_samples"][0]["audio"]["sample_rate"] == 22050


class TestSpeakerRecognitionClientRecognize:
    """Tests for recognition endpoint."""

    @pytest.mark.asyncio
    async def test_recognize_success(self) -> None:
        """Test successful recognition request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "alice",
            "confidence": 0.95,
            "all_scores": {"alice": 0.95, "bob": 0.45},
        }

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            client = SpeakerRecognitionClient("http://localhost:8099")
            request = RecognitionRequest(
                audio=AudioInput(audio_data="SGVsbG8=", sample_rate=16000)
            )
            result = await client.recognize(request)

            assert isinstance(result, RecognitionResult)
            assert result.user_id == "alice"
            assert result.confidence == 0.95
            assert result.all_scores["alice"] == 0.95
            assert result.all_scores["bob"] == 0.45
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_recognize_sends_correct_payload(self) -> None:
        """Test that recognition sends correct JSON payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "bob",
            "confidence": 0.88,
            "all_scores": {"bob": 0.88},
        }

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            client = SpeakerRecognitionClient("http://localhost:8099")
            request = RecognitionRequest(
                audio=AudioInput(audio_data="YXVkaW8=", sample_rate=44100)
            )
            await client.recognize(request)

            call_kwargs = mock_post.call_args
            assert call_kwargs[0][0] == "/recognize"
            json_payload = call_kwargs[1]["json"]
            assert "audio" in json_payload
            assert json_payload["audio"]["audio_data"] == "YXVkaW8="
            assert json_payload["audio"]["sample_rate"] == 44100


class TestSpeakerRecognitionClientEnsureClient:
    """Tests for _ensure_client method."""

    @pytest.mark.asyncio
    async def test_ensure_client_creates_client_if_none(self) -> None:
        """Test that _ensure_client creates client if not exists."""
        client = SpeakerRecognitionClient("http://localhost:8099")
        assert client._client is None

        internal_client = client._ensure_client()
        assert internal_client is not None
        assert isinstance(internal_client, httpx.AsyncClient)
        assert client._client is internal_client

        # Cleanup
        await client.close()

    @pytest.mark.asyncio
    async def test_ensure_client_returns_existing_client(self) -> None:
        """Test that _ensure_client returns existing client."""
        client = SpeakerRecognitionClient("http://localhost:8099")
        first_client = client._ensure_client()
        second_client = client._ensure_client()

        assert first_client is second_client

        # Cleanup
        await client.close()


class TestSyncSpeakerRecognitionClient:
    """Tests for synchronous client."""

    def test_sync_client_init(self) -> None:
        """Test sync client initialization."""
        client = SyncSpeakerRecognitionClient(base_url="http://localhost:8099")
        assert client._base_url == "http://localhost:8099"
        assert client._timeout == 30.0
        assert client._client is None

    def test_sync_client_context_manager(self) -> None:
        """Test sync client context manager."""
        with SyncSpeakerRecognitionClient("http://localhost:8099") as client:
            assert client._client is not None
            assert isinstance(client._client, httpx.Client)

    def test_sync_client_ensure_client(self) -> None:
        """Test sync client _ensure_client method."""
        client = SyncSpeakerRecognitionClient("http://localhost:8099")
        internal_client = client._ensure_client()
        assert internal_client is not None
        assert isinstance(internal_client, httpx.Client)
        client.close()

    def test_sync_client_health_check(self) -> None:
        """Test sync client health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch.object(httpx.Client, "get") as mock_get:
            mock_get.return_value = mock_response

            client = SyncSpeakerRecognitionClient("http://localhost:8099")
            result = client.health_check()

            assert isinstance(result, HealthResponse)
            assert result.status == "healthy"
            client.close()

    def test_sync_client_train(self) -> None:
        """Test sync client training."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "trained_users": ["alice"],
            "count": 1,
        }

        with patch.object(httpx.Client, "post") as mock_post:
            mock_post.return_value = mock_response

            client = SyncSpeakerRecognitionClient("http://localhost:8099")
            request = TrainingRequest(
                voice_samples=[
                    VoiceSample(
                        user="alice",
                        audio=AudioInput(audio_data="SGVsbG8="),
                    )
                ]
            )
            result = client.train(request)

            assert isinstance(result, TrainingResult)
            assert result.status == "success"
            client.close()

    def test_sync_client_recognize(self) -> None:
        """Test sync client recognition."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "alice",
            "confidence": 0.92,
            "all_scores": {"alice": 0.92},
        }

        with patch.object(httpx.Client, "post") as mock_post:
            mock_post.return_value = mock_response

            client = SyncSpeakerRecognitionClient("http://localhost:8099")
            request = RecognitionRequest(
                audio=AudioInput(audio_data="SGVsbG8=")
            )
            result = client.recognize(request)

            assert isinstance(result, RecognitionResult)
            assert result.user_id == "alice"
            assert result.confidence == 0.92
            client.close()


class TestClientErrorHandling:
    """Tests for client error handling."""

    @pytest.mark.asyncio
    async def test_health_check_raises_on_http_error(self) -> None:
        """Test that HTTP errors are raised."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            client = SpeakerRecognitionClient("http://localhost:8099")
            with pytest.raises(httpx.HTTPStatusError):
                await client.health_check()

    @pytest.mark.asyncio
    async def test_train_raises_on_http_error(self) -> None:
        """Test that training raises on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            client = SpeakerRecognitionClient("http://localhost:8099")
            request = TrainingRequest(voice_samples=[])
            with pytest.raises(httpx.HTTPStatusError):
                await client.train(request)

    @pytest.mark.asyncio
    async def test_recognize_raises_on_http_error(self) -> None:
        """Test that recognition raises on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Model not trained",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            client = SpeakerRecognitionClient("http://localhost:8099")
            request = RecognitionRequest(audio=AudioInput(audio_data="SGVsbG8="))
            with pytest.raises(httpx.HTTPStatusError):
                await client.recognize(request)
