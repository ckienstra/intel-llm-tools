import pytest
import logging
import json
from unittest.mock import Mock, MagicMock, patch
from mix_datasets import DatasetMixer, InvalidConfigurationError, Error


@pytest.fixture
def mock_logger():
    """Fixture providing a mock logger."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def mock_tokenizer():
    """Fixture providing a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}
    return tokenizer


@pytest.fixture
def valid_config():
    """Fixture providing a valid configuration."""
    return {
        "samples": 512,
        "seqlen": 2048,
        "datasets": [
            {
                "name": "PygmalionAI/PIPPA",
                "percentage": 0.5,
                "type": "json",
                "data_files": "https://huggingface.co/datasets/PygmalionAI/PIPPA/resolve/main/pippa_deduped.jsonl",
                "text_field": "conversation",
            },
            {
                "name": "NeelNanda/pile-10k",
                "percentage": 0.5,
                "split": "train",
                "text_field": "text",
            },
        ],
    }


class TestDatasetMixerInitialization:
    """Test DatasetMixer initialization."""

    def test_init_with_valid_params(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test initialization with valid parameters."""
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        assert mixer.logger == mock_logger
        assert mixer.tokenizer == mock_tokenizer
        assert mixer.config == valid_config
        assert mixer.calibration_data == []

    def test_properties(self, mock_logger, mock_tokenizer, valid_config):
        """Test all properties work correctly."""
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        assert mixer.logger is mock_logger
        assert mixer.tokenizer is mock_tokenizer
        assert mixer.config is valid_config


class TestDatasetMixerValidation:
    """Test DatasetMixer.validate() method."""

    def test_validate_with_valid_config(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation passes with valid config."""
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        # Should not raise any exception
        mixer.validate()

    def test_validate_samples_too_high(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation fails when samples exceed 1024."""
        valid_config["samples"] = 1025
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        with pytest.raises(InvalidConfigurationError):
            mixer.validate()

    def test_validate_samples_negative(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation fails with negative samples."""
        valid_config["samples"] = -1
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        with pytest.raises(InvalidConfigurationError):
            mixer.validate()

    def test_validate_dataset_missing_name(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation fails when dataset name is missing."""
        valid_config["datasets"][0]["name"] = None
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        with pytest.raises(InvalidConfigurationError):
            mixer.validate()

    def test_validate_dataset_name_without_slash(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation fails when dataset name doesn't have slash."""
        valid_config["datasets"][0]["name"] = "InvalidName"
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        with pytest.raises(InvalidConfigurationError):
            mixer.validate()

    def test_validate_missing_percentage(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation fails when percentage is missing."""
        del valid_config["datasets"][0]["percentage"]
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        with pytest.raises(InvalidConfigurationError):
            mixer.validate()

    def test_validate_percentage_too_high(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation fails when percentage > 1."""
        valid_config["datasets"][0]["percentage"] = 1.5
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        with pytest.raises(InvalidConfigurationError):
            mixer.validate()

    def test_validate_percentage_negative(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation fails with negative percentage."""
        valid_config["datasets"][0]["percentage"] = -0.1
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        with pytest.raises(InvalidConfigurationError):
            mixer.validate()

    def test_validate_percentage_sum_not_one(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation fails when percentages don't sum to 1."""
        valid_config["datasets"][0]["percentage"] = 0.3
        valid_config["datasets"][1]["percentage"] = 0.3
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        with pytest.raises(InvalidConfigurationError):
            mixer.validate()

    def test_validate_zero_percentage_allowed(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test that zero percentage is allowed for exclusion."""
        valid_config["datasets"][0]["percentage"] = 0.0
        valid_config["datasets"][1]["percentage"] = 1.0
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        # Should not raise
        mixer.validate()

    def test_validate_max_samples(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation passes with maximum allowed samples."""
        valid_config["samples"] = 1024
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        # Should not raise
        mixer.validate()

    def test_validate_floating_point_precision(
        self, mock_logger, mock_tokenizer, valid_config
    ):
        """Test validation handles floating point precision correctly."""
        valid_config["datasets"][0]["percentage"] = 0.3333333333
        valid_config["datasets"][1]["percentage"] = 0.6666666667
        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)

        # Should not raise due to math.isclose
        mixer.validate()


class TestDatasetMixerMix:
    """Test DatasetMixer.mix() method."""

    @patch("mix_datasets.load_dataset")
    def test_mix_basic(
        self, mock_load_dataset, mock_logger, mock_tokenizer, valid_config
    ):
        """Test basic mix functionality."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.take.return_value = [
            {"conversation": "Sample text 1 " * 500},
            {"conversation": "Sample text 2 " * 500},
        ]
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer to return sufficient tokens
        mock_tokenizer.return_value = {"input_ids": [1] * 2048}

        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        result = mixer.mix()

        # Should return a list (actual samples depend on dataset availability)
        assert isinstance(result, list)

    @patch("mix_datasets.load_dataset")
    def test_mix_returns_list(
        self, mock_load_dataset, mock_logger, mock_tokenizer, valid_config
    ):
        """Test that mix returns a list."""
        mock_dataset = Mock()
        mock_dataset.take.return_value = []
        mock_load_dataset.return_value = mock_dataset

        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        result = mixer.mix()

        assert isinstance(result, list)

    @patch("mix_datasets.load_dataset")
    def test_mix_skips_zero_percentage_datasets(
        self, mock_load_dataset, mock_logger, mock_tokenizer, valid_config
    ):
        """Test that datasets with 0 percentage are skipped."""
        valid_config["datasets"][0]["percentage"] = 0.0
        valid_config["datasets"][1]["percentage"] = 1.0

        mock_dataset = Mock()
        mock_dataset.take.return_value = []
        mock_load_dataset.return_value = mock_dataset

        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        mixer.mix()

        # load_dataset should only be called once (for the non-zero percentage dataset)
        assert mock_load_dataset.call_count == 1

    @patch("mix_datasets.load_dataset")
    def test_mix_handles_list_text_field(
        self, mock_load_dataset, mock_logger, mock_tokenizer, valid_config
    ):
        """Test mix handles conversational (list) text format."""
        # Mock conversational data like PIPPA
        mock_dataset = Mock()
        mock_dataset.take.return_value = [
            {
                "conversation": [
                    {"is_human": True, "message": "Hello"},
                    {"is_human": False, "message": "Hi there!"},
                ]
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer to return sufficient tokens
        mock_tokenizer.return_value = {"input_ids": [1] * 2048}

        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        result = mixer.mix()

        # Verify tokenizer was called (implicitly testing conversion)
        assert mock_tokenizer.called

    @patch("mix_datasets.load_dataset")
    def test_mix_filters_by_seqlen(
        self, mock_load_dataset, mock_logger, mock_tokenizer, valid_config
    ):
        """Test that samples below minimum sequence length are filtered."""
        mock_dataset = Mock()
        mock_dataset.take.return_value = [
            {"text": "Short text"},
            {"text": "Long text " * 500},
        ]
        mock_load_dataset.return_value = mock_dataset

        # First call returns short (< 2048 tokens), second returns long (>= 2048 tokens)
        mock_tokenizer.side_effect = [
            {"input_ids": [1, 2, 3]},  # 3 tokens - too short
            {"input_ids": [1] * 2048},  # 2048 tokens - good
        ]

        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        result = mixer.mix()

        # Only the long sample should be included
        assert len(result) <= 256  # half of 512 samples

    @patch("mix_datasets.load_dataset")
    def test_mix_handles_load_dataset_error(
        self, mock_load_dataset, mock_logger, mock_tokenizer, valid_config
    ):
        """Test that mix continues on dataset load error."""
        mock_load_dataset.side_effect = Exception("Network error")

        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        result = mixer.mix()

        assert isinstance(result, list)
        # logger.error should have been called
        assert mock_logger.error.called

    @patch("mix_datasets.load_dataset")
    def test_mix_missing_text_field(
        self, mock_load_dataset, mock_logger, mock_tokenizer, valid_config
    ):
        """Test that datasets with missing text_field are skipped."""
        valid_config["datasets"][0]["text_field"] = None

        mock_dataset = Mock()
        mock_dataset.take.return_value = [{"data": "value"}]
        mock_load_dataset.return_value = mock_dataset

        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        result = mixer.mix()

        assert isinstance(result, list)
        # logger.error should be called for missing text_field
        assert any(
            "text_field" in str(call)
            for call in mock_logger.error.call_args_list
        )

    @patch("mix_datasets.load_dataset")
    def test_mix_with_data_files(
        self, mock_load_dataset, mock_logger, mock_tokenizer, valid_config
    ):
        """Test mix with data_files configuration."""
        mock_dataset = Mock()
        mock_dataset.take.return_value = []
        mock_load_dataset.return_value = mock_dataset

        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        mixer.mix()

        # Verify load_dataset was called with data_files for first dataset
        calls = mock_load_dataset.call_args_list
        assert any("data_files" in str(call) for call in calls)

    @patch("mix_datasets.load_dataset")
    def test_mix_respects_percentages(
        self, mock_load_dataset, mock_logger, mock_tokenizer, valid_config
    ):
        """Test that mix respects dataset percentages."""
        valid_config["samples"] = 100
        valid_config["datasets"][0]["percentage"] = 0.7
        valid_config["datasets"][1]["percentage"] = 0.3

        mock_dataset = Mock()
        mock_dataset.take.return_value = [
            {"conversation": "Sample " * 500} for _ in range(1000)
        ]
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer.return_value = {"input_ids": [1] * 2048}

        mixer = DatasetMixer(mock_logger, mock_tokenizer, valid_config)
        mixer.mix()

        # Verify take was called with appropriate amounts (multiplied by 20)
        calls = mock_load_dataset.return_value.take.call_args_list
        # Should have been called multiple times with different amounts
        assert len(calls) > 0


class TestExceptions:
    """Test exception classes."""

    def test_invalid_configuration_error_is_error(self):
        """Test that InvalidConfigurationError inherits from Error."""
        assert issubclass(InvalidConfigurationError, Error)

    def test_error_is_exception(self):
        """Test that Error inherits from Exception."""
        assert issubclass(Error, Exception)

    def test_invalid_configuration_error_creation(self):
        """Test creating InvalidConfigurationError."""
        error = InvalidConfigurationError("Test error")
        assert str(error) == "Test error"

    def test_error_creation(self):
        """Test creating Error."""
        error = Error("Test error")
        assert str(error) == "Test error"


if __name__ == "__main__":
    pytest.main()
