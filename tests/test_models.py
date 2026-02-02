"""Unit tests for speaker recognition models."""

import pytest
from pydantic import ValidationError

from speaker_recognition.models import (
    AudioInput,
    Config,
    ErrorResponse,
    HealthResponse,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
    VoiceSample,
)


class TestAudioInput:
    """Tests for AudioInput model."""

    def test_valid_audio_input(self) -> None:
        """Test creating a valid AudioInput."""
        audio = AudioInput(audio_data="SGVsbG8gV29ybGQ=", sample_rate=16000)
        assert audio.audio_data == "SGVsbG8gV29ybGQ="
        assert audio.sample_rate == 16000

    def test_default_sample_rate(self) -> None:
        """Test default sample rate of 16000."""
        audio = AudioInput(audio_data="SGVsbG8gV29ybGQ=")
        assert audio.sample_rate == 16000

    def test_custom_sample_rate(self) -> None:
        """Test custom sample rate."""
        audio = AudioInput(audio_data="SGVsbG8gV29ybGQ=", sample_rate=44100)
        assert audio.sample_rate == 44100

    def test_missing_audio_data_raises_error(self) -> None:
        """Test that missing audio_data raises ValidationError."""
        with pytest.raises(ValidationError):
            AudioInput()  # type: ignore[call-arg]


class TestVoiceSample:
    """Tests for VoiceSample model."""

    def test_valid_voice_sample(self) -> None:
        """Test creating a valid VoiceSample."""
        audio = AudioInput(audio_data="SGVsbG8=", sample_rate=16000)
        sample = VoiceSample(user="alice", audio=audio)
        assert sample.user == "alice"
        assert sample.audio.audio_data == "SGVsbG8="

    def test_missing_user_raises_error(self) -> None:
        """Test that missing user raises ValidationError."""
        audio = AudioInput(audio_data="SGVsbG8=")
        with pytest.raises(ValidationError):
            VoiceSample(audio=audio)  # type: ignore[call-arg]

    def test_missing_audio_raises_error(self) -> None:
        """Test that missing audio raises ValidationError."""
        with pytest.raises(ValidationError):
            VoiceSample(user="alice")  # type: ignore[call-arg]


class TestTrainingRequest:
    """Tests for TrainingRequest model."""

    def test_valid_training_request(self) -> None:
        """Test creating a valid TrainingRequest."""
        samples = [
            VoiceSample(
                user="alice",
                audio=AudioInput(audio_data="SGVsbG8=", sample_rate=16000),
            ),
            VoiceSample(
                user="bob",
                audio=AudioInput(audio_data="V29ybGQ=", sample_rate=16000),
            ),
        ]
        request = TrainingRequest(voice_samples=samples)
        assert len(request.voice_samples) == 2
        assert request.voice_samples[0].user == "alice"
        assert request.voice_samples[1].user == "bob"

    def test_empty_voice_samples(self) -> None:
        """Test creating a request with empty voice samples."""
        request = TrainingRequest(voice_samples=[])
        assert len(request.voice_samples) == 0

    def test_missing_voice_samples_raises_error(self) -> None:
        """Test that missing voice_samples raises ValidationError."""
        with pytest.raises(ValidationError):
            TrainingRequest()  # type: ignore[call-arg]


class TestTrainingResult:
    """Tests for TrainingResult model."""

    def test_valid_training_result(self) -> None:
        """Test creating a valid TrainingResult."""
        result = TrainingResult(
            status="success",
            trained_users=["alice", "bob"],
            count=2,
        )
        assert result.status == "success"
        assert result.trained_users == ["alice", "bob"]
        assert result.count == 2

    def test_training_result_empty_users(self) -> None:
        """Test creating a result with no trained users."""
        result = TrainingResult(
            status="partial",
            trained_users=[],
            count=0,
        )
        assert result.count == 0
        assert len(result.trained_users) == 0


class TestRecognitionRequest:
    """Tests for RecognitionRequest model."""

    def test_valid_recognition_request(self) -> None:
        """Test creating a valid RecognitionRequest."""
        audio = AudioInput(audio_data="SGVsbG8=", sample_rate=16000)
        request = RecognitionRequest(audio=audio)
        assert request.audio.audio_data == "SGVsbG8="
        assert request.audio.sample_rate == 16000

    def test_missing_audio_raises_error(self) -> None:
        """Test that missing audio raises ValidationError."""
        with pytest.raises(ValidationError):
            RecognitionRequest()  # type: ignore[call-arg]


class TestRecognitionResult:
    """Tests for RecognitionResult model."""

    def test_valid_recognition_result(self) -> None:
        """Test creating a valid RecognitionResult."""
        result = RecognitionResult(
            user_id="alice",
            confidence=0.95,
            all_scores={"alice": 0.95, "bob": 0.45},
        )
        assert result.user_id == "alice"
        assert result.confidence == 0.95
        assert result.all_scores["alice"] == 0.95
        assert result.all_scores["bob"] == 0.45

    def test_recognition_result_with_single_user(self) -> None:
        """Test result with a single user."""
        result = RecognitionResult(
            user_id="alice",
            confidence=0.8,
            all_scores={"alice": 0.8},
        )
        assert result.user_id == "alice"
        assert len(result.all_scores) == 1


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_valid_health_response(self) -> None:
        """Test creating a valid HealthResponse."""
        response = HealthResponse(status="healthy")
        assert response.status == "healthy"

    def test_unhealthy_response(self) -> None:
        """Test creating an unhealthy response."""
        response = HealthResponse(status="unhealthy")
        assert response.status == "unhealthy"


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_valid_error_response(self) -> None:
        """Test creating a valid ErrorResponse."""
        response = ErrorResponse(error="Something went wrong")
        assert response.error == "Something went wrong"


class TestConfig:
    """Tests for Config model."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Config()
        assert config.host == "0.0.0.0"
        assert config.port == 8099
        assert config.log_level == "INFO"
        assert config.access_log is True
        assert config.embeddings_directory == "./embeddings"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = Config(
            host="127.0.0.1",
            port=9000,
            log_level="DEBUG",
            access_log=False,
            embeddings_directory="/custom/path",
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.log_level == "DEBUG"
        assert config.access_log is False
        assert config.embeddings_directory == "/custom/path"


class TestModelSerialization:
    """Tests for model serialization/deserialization."""

    def test_training_request_json_roundtrip(self) -> None:
        """Test JSON serialization and deserialization of TrainingRequest."""
        original = TrainingRequest(
            voice_samples=[
                VoiceSample(
                    user="alice",
                    audio=AudioInput(audio_data="SGVsbG8=", sample_rate=16000),
                )
            ]
        )
        json_data = original.model_dump_json()
        restored = TrainingRequest.model_validate_json(json_data)
        assert restored.voice_samples[0].user == original.voice_samples[0].user
        assert (
            restored.voice_samples[0].audio.audio_data
            == original.voice_samples[0].audio.audio_data
        )

    def test_recognition_result_json_roundtrip(self) -> None:
        """Test JSON serialization and deserialization of RecognitionResult."""
        original = RecognitionResult(
            user_id="alice",
            confidence=0.92,
            all_scores={"alice": 0.92, "bob": 0.35, "charlie": 0.12},
        )
        json_data = original.model_dump_json()
        restored = RecognitionResult.model_validate_json(json_data)
        assert restored.user_id == original.user_id
        assert restored.confidence == original.confidence
        assert restored.all_scores == original.all_scores
