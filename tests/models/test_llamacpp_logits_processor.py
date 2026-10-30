"""llama-cpp-python passes the sampling parameters to its logits processors as
positional arguments, the Outlines ones take them as keyword arguments. These
tests use a tiny stand-in for `llama_cpp` so they do not need a model file."""

import sys
import types

import numpy as np
import pytest

from outlines.processors import OutlinesLogitsProcessor


class LlamaLogitsProcessor:
    """Copy of the wrapper `llama_cpp.LogitsProcessorList` builds around a
    processor, see llama_cpp/_logits_processor.py in llama-cpp-python."""

    def __init__(self, logits_processor, **kwargs):
        self.logits_processor = logits_processor
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_ids, scores):
        return self.logits_processor(input_ids, scores)


class LogitsProcessorList(list):
    def __getitem__(self, index):
        processor = list.__getitem__(self, index)
        if isinstance(index, slice):
            return processor
        if isinstance(processor, LlamaLogitsProcessor):
            return processor
        wrapped = LlamaLogitsProcessor(
            processor,
            thought_enabled_kwargs=("skip_prompt", "skip_prompt_stop_str"),
            core_kwargs=("top_k", "top_p", "min_p", "temp", "logit_bias"),
        )
        list.__setitem__(self, index, wrapped)
        return wrapped


class RecordingProcessor(OutlinesLogitsProcessor):
    """Stands in for an Outlines processor, which takes the sampling
    parameters as keyword arguments."""

    def __init__(self):
        super().__init__(tensor_library_name="numpy")
        self.calls: list = []

    def __call__(self, input_ids, logits, **params):
        self.calls.append(params)
        return logits

    def process_logits(self, input_ids, logits):  # pragma: no cover
        return logits


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    module = types.ModuleType("llama_cpp")
    module.LogitsProcessorList = LogitsProcessorList
    module.LlamaLogitsProcessor = LlamaLogitsProcessor
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return module


def _generate(adapter, processor, **inference_kwargs):
    """Run the processor list the way llama-cpp-python does for one token."""
    processor_list = adapter.format_output_type(processor, **inference_kwargs)
    input_ids = np.array([[1, 2, 3]])
    scores = np.full((1, 5), -10.0, dtype=np.float32)
    for item in processor_list:
        scores = item(input_ids, scores)
    return scores


def test_sampling_params_reach_the_processor(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    processor = RecordingProcessor()
    _generate(LlamaCppTypeAdapter(), processor, top_k=5, temp=0.2, max_tokens=32)

    assert len(processor.calls) == 1
    assert processor.calls[0] == {"top_k": 5, "temp": 0.2, "max_tokens": 32}


def test_no_sampling_params(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    processor = RecordingProcessor()
    _generate(LlamaCppTypeAdapter(), processor)

    assert processor.calls == [{}]


def test_stream_is_not_forwarded(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    processor = RecordingProcessor()
    # `stream` is a generation flag, not a sampling parameter, and
    # llama-cpp-python does not pass it to the processors.
    _generate(LlamaCppTypeAdapter(), processor, temp=0.0)

    assert processor.calls[0] == {"temp": 0.0}


def test_no_output_type_returns_none(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    assert LlamaCppTypeAdapter().format_output_type(None) is None


def test_processor_output_is_unchanged(fake_llama_cpp):
    from outlines.models.llamacpp import LlamaCppTypeAdapter

    guided = _generate(LlamaCppTypeAdapter(), RecordingProcessor(), top_k=1)

    np.testing.assert_array_equal(guided, np.full((1, 5), -10.0, dtype=np.float32))
