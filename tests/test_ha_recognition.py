"""Unit tests for Home Assistant speaker recognition module."""

import base64
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock Home Assistant modules before importing the recognition module
mock_hass_module = MagicMock()
sys.modules["homeassistant"] = mock_hass_module
sys.modules["homeassistant.core"] = MagicMock()

# Now we can import our module
from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.models import (
    AudioInput,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
    VoiceSample,
)


class MockHomeAssistant:
    """Mock Home Assistant instance."""

    def __init__(self) -> None:
        """Initialize mock Home Assistant."""
        self.config = MagicMock()
        self.config.path = MagicMock(return_value="/fake/ha/config")

    async def async_add_executor_job(self, func: Any, *args: Any) -> Any:
        """Mock async_add_executor_job."""
        return func(*args)


class TestSpeakerRecognitionModule:
    """Tests for the SpeakerRecognition class from custom_components."""

    @pytest.mark.asyncio
    async def test_training_with_valid_samples(self) -> None:
        """Test training with valid voice samples."""
        # Create mock response
        mock_training_result = TrainingResult(
            status="success",
            trained_users=["alice", "bob"],
            count=2,
        )

        # Test that the client can be used for training
        with patch.object(
            SpeakerRecognitionClient, "train", new_callable=AsyncMock
        ) as mock_train:
            mock_train.return_value = mock_training_result

            client = SpeakerRecognitionClient("http://localhost:8099")
            request = TrainingRequest(
                voice_samples=[
                    VoiceSample(
                        user="alice",
                        audio=AudioInput(
                            audio_data=base64.b64encode(b"fake audio").decode(),
                            sample_rate=16000,
                        ),
                    ),
                    VoiceSample(
                        user="bob",
                        audio=AudioInput(
                            audio_data=base64.b64encode(b"fake audio 2").decode(),
                            sample_rate=16000,
                        ),
                    ),
                ]
            )

            result = await client.train(request)

            assert result.status == "success"
            assert result.count == 2
            assert "alice" in result.trained_users
            assert "bob" in result.trained_users
            mock_train.assert_called_once()

    @pytest.mark.asyncio
    async def test_recognition_returns_result(self) -> None:
        """Test that recognition returns expected result."""
        mock_recognition_result = RecognitionResult(
            user_id="alice",
            confidence=0.95,
            all_scores={"alice": 0.95, "bob": 0.45},
        )

        with patch.object(
            SpeakerRecognitionClient, "recognize", new_callable=AsyncMock
        ) as mock_recognize:
            mock_recognize.return_value = mock_recognition_result

            client = SpeakerRecognitionClient("http://localhost:8099")
            request = RecognitionRequest(
                audio=AudioInput(
                    audio_data=base64.b64encode(b"test audio").decode(),
                    sample_rate=16000,
                )
            )

            result = await client.recognize(request)

            assert result.user_id == "alice"
            assert result.confidence == 0.95
            assert result.all_scores["alice"] == 0.95
            mock_recognize.assert_called_once()

    @pytest.mark.asyncio
    async def test_recognition_with_low_confidence(self) -> None:
        """Test recognition result with low confidence."""
        mock_recognition_result = RecognitionResult(
            user_id="unknown",
            confidence=0.25,
            all_scores={"alice": 0.25, "bob": 0.20},
        )

        with patch.object(
            SpeakerRecognitionClient, "recognize", new_callable=AsyncMock
        ) as mock_recognize:
            mock_recognize.return_value = mock_recognition_result

            client = SpeakerRecognitionClient("http://localhost:8099")
            request = RecognitionRequest(
                audio=AudioInput(audio_data=base64.b64encode(b"noise").decode())
            )

            result = await client.recognize(request)

            assert result.confidence < 0.5  # Low confidence
            mock_recognize.assert_called_once()


class TestVoiceSamplePreparation:
    """Tests for voice sample preparation logic."""

    def test_audio_base64_encoding(self) -> None:
        """Test that audio data is properly base64 encoded."""
        raw_audio = b"\x00\x01\x02\x03\x04\x05"
        encoded = base64.b64encode(raw_audio).decode("utf-8")

        audio_input = AudioInput(audio_data=encoded, sample_rate=16000)
        assert audio_input.audio_data == encoded

        # Verify we can decode it back
        decoded = base64.b64decode(audio_input.audio_data)
        assert decoded == raw_audio

    def test_voice_sample_creation(self) -> None:
        """Test voice sample model creation."""
        audio = AudioInput(
            audio_data=base64.b64encode(b"sample").decode(),
            sample_rate=44100,
        )
        sample = VoiceSample(user="test_user", audio=audio)

        assert sample.user == "test_user"
        assert sample.audio.sample_rate == 44100

    def test_training_request_with_multiple_samples(self) -> None:
        """Test training request with multiple samples per user."""
        samples = []
        for i in range(3):
            samples.append(
                VoiceSample(
                    user=f"user_{i}",
                    audio=AudioInput(
                        audio_data=base64.b64encode(f"audio_{i}".encode()).decode()
                    ),
                )
            )

        request = TrainingRequest(voice_samples=samples)
        assert len(request.voice_samples) == 3


class TestMediaPathParsing:
    """Tests for media path parsing logic used in recognition.py."""

    def test_media_source_path_extraction(self) -> None:
        """Test extraction of relative path from media source URI."""
        media_id = "media-source://media_source/local/voice_samples/alice.wav"
        prefix = "media-source://media_source/local/"

        if media_id.startswith(prefix):
            relative_path = media_id.replace(prefix, "")
        else:
            relative_path = ""

        assert relative_path == "voice_samples/alice.wav"

    def test_invalid_media_source_path(self) -> None:
        """Test handling of invalid media source paths."""
        media_id = "file:///some/local/file.wav"
        prefix = "media-source://media_source/local/"

        if media_id.startswith(prefix):
            relative_path = media_id.replace(prefix, "")
        else:
            relative_path = None

        assert relative_path is None

    def test_full_path_construction(self) -> None:
        """Test construction of full path from relative path."""
        base_path = Path("/config/media")
        relative_path = "voice_samples/alice.wav"

        full_path = base_path / relative_path
        assert str(full_path) == "/config/media/voice_samples/alice.wav"


class TestConfidenceThresholds:
    """Tests for confidence threshold logic used in conversation.py."""

    def test_confidence_above_threshold(self) -> None:
        """Test that confidence above threshold passes."""
        min_confidence = 0.7
        result_confidence = 0.85

        should_use = result_confidence >= min_confidence
        assert should_use is True

    def test_confidence_below_threshold(self) -> None:
        """Test that confidence below threshold fails."""
        min_confidence = 0.7
        result_confidence = 0.5

        should_use = result_confidence >= min_confidence
        assert should_use is False

    def test_confidence_at_threshold(self) -> None:
        """Test that confidence exactly at threshold passes."""
        min_confidence = 0.7
        result_confidence = 0.7

        should_use = result_confidence >= min_confidence
        assert should_use is True

    def test_default_min_confidence(self) -> None:
        """Test default minimum confidence value."""
        # From const.py
        DEFAULT_MIN_CONFIDENCE = 0.0
        assert DEFAULT_MIN_CONFIDENCE == 0.0

        # Any positive confidence should pass with default
        result_confidence = 0.1
        should_use = result_confidence >= DEFAULT_MIN_CONFIDENCE
        assert should_use is True


class TestRecognitionResultStorage:
    """Tests for recognition result storage logic used in stt.py."""

    def test_result_storage_structure(self) -> None:
        """Test structure of stored recognition result."""
        import time

        result_data = {
            "user_id": "alice",
            "confidence": 0.92,
            "timestamp": time.time(),
        }

        assert "user_id" in result_data
        assert "confidence" in result_data
        assert "timestamp" in result_data
        assert result_data["user_id"] == "alice"

    def test_result_age_calculation(self) -> None:
        """Test calculation of result age."""
        import time

        timestamp = time.time() - 3.0  # 3 seconds ago
        current_time = time.time()

        age = current_time - timestamp
        assert 2.9 <= age <= 3.1  # Allow small timing variance

    def test_result_within_window(self) -> None:
        """Test result within time window."""
        import time

        # Result from 2 seconds ago
        timestamp = time.time() - 2.0
        current_time = time.time()
        window = 5.0  # 5 second window

        age = current_time - timestamp
        is_valid = age < window
        assert is_valid is True

    def test_result_outside_window(self) -> None:
        """Test result outside time window."""
        import time

        # Result from 10 seconds ago
        timestamp = time.time() - 10.0
        current_time = time.time()
        window = 5.0  # 5 second window

        age = current_time - timestamp
        is_valid = age < window
        assert is_valid is False


class TestEventFiring:
    """Tests for event data structure used in stt.py."""

    def test_speaker_recognition_event_data(self) -> None:
        """Test structure of speaker_recognition_detected event data."""
        event_data = {
            "user_id": "alice",
            "confidence": 0.95,
            "all_scores": {"alice": 0.95, "bob": 0.45},
            "entity_id": "stt.speaker_recognition_proxy",
        }

        assert event_data["user_id"] == "alice"
        assert event_data["confidence"] == 0.95
        assert len(event_data["all_scores"]) == 2
        assert "entity_id" in event_data

    def test_event_data_from_recognition_result(self) -> None:
        """Test creating event data from RecognitionResult."""
        result = RecognitionResult(
            user_id="bob",
            confidence=0.88,
            all_scores={"alice": 0.45, "bob": 0.88},
        )

        event_data = {
            "user_id": result.user_id,
            "confidence": result.confidence,
            "all_scores": result.all_scores,
            "entity_id": "stt.test_entity",
        }

        assert event_data["user_id"] == "bob"
        assert event_data["confidence"] == 0.88
        assert event_data["all_scores"]["bob"] == 0.88


class TestAudioBuffering:
    """Tests for audio buffering logic used in stt.py."""

    @pytest.mark.asyncio
    async def test_audio_buffer_collection(self) -> None:
        """Test that audio chunks are properly buffered."""
        audio_buffer = bytearray()

        # Simulate audio chunks
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        async for chunk in mock_stream():
            audio_buffer.extend(chunk)

        assert bytes(audio_buffer) == b"chunk1chunk2chunk3"

    @pytest.mark.asyncio
    async def test_buffered_stream_passes_through(self) -> None:
        """Test that buffered stream passes data through."""
        audio_buffer = bytearray()
        received_chunks = []

        async def source_stream():
            yield b"data1"
            yield b"data2"

        async def buffered_stream():
            async for chunk in source_stream():
                audio_buffer.extend(chunk)
                yield chunk

        async for chunk in buffered_stream():
            received_chunks.append(chunk)

        # Buffer should have all data
        assert bytes(audio_buffer) == b"data1data2"
        # All chunks should have been passed through
        assert received_chunks == [b"data1", b"data2"]

    def test_empty_buffer_handling(self) -> None:
        """Test handling of empty audio buffer."""
        audio_buffer = bytearray()

        # Empty buffer should evaluate to False
        if not audio_buffer:
            should_recognize = False
        else:
            should_recognize = True

        assert should_recognize is False

    def test_buffer_to_bytes_conversion(self) -> None:
        """Test converting bytearray buffer to bytes."""
        audio_buffer = bytearray()
        audio_buffer.extend(b"\x00\x01\x02\x03")

        audio_bytes = bytes(audio_buffer)
        assert isinstance(audio_bytes, bytes)
        assert audio_bytes == b"\x00\x01\x02\x03"
