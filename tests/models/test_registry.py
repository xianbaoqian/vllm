# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import warnings
from types import SimpleNamespace

import pytest
import torch.cuda

from vllm.logger import _print_warning_once
from vllm.model_executor.models import (
    is_pooling_model,
    is_text_generation_model,
    supports_multimodal,
)
from vllm.model_executor.models.adapters import (
    as_embedding_model,
    as_seq_cls_model,
)
from vllm.model_executor.models.registry import (
    _MULTIMODAL_MODELS,
    _OOT_SUPPORTED_MODELS,
    _PREVIOUSLY_SUPPORTED_MODELS,
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
    ModelRegistry,
    _LazyRegisteredModel,
)
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test
from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", ModelRegistry.get_supported_archs())
def test_registry_imports(model_arch):
    # Skip if transformers version is incompatible
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )

    if model_arch in ("PrithviGeoSpatialMAE", "Terratorch"):
        import importlib.util

        if importlib.util.find_spec("terratorch") is None:
            pytest.skip(
                "terratorch is not installed; "
                "temporarily skipped while PyPI has `lightning` quarantined "
                "(see #41376)"
            )

    # DSpark draft model is supported on CUDA and ROCm; stubbed to None on XPU.
    if model_arch == "DSparkDraftModel" and not (
        current_platform.is_cuda() or current_platform.is_rocm()
    ):
        pytest.skip("DSparkDraftModel is only supported on CUDA and ROCm")

    if model_arch in ("Dots3NoteForCausalLM", "Dots3NoteMTPModel") and not (
        current_platform.is_cuda()
    ):
        pytest.skip("Dots3 NOTE is only supported on CUDA")

    # Ensure all model classes can be imported successfully
    model_cls = ModelRegistry._try_load_model_cls(model_arch)
    assert model_cls is not None

    if model_arch in _SPECULATIVE_DECODING_MODELS:
        return  # Ignore these models which do not have a unified format

    if model_arch in _TEXT_GENERATION_MODELS or model_arch in _MULTIMODAL_MODELS:
        assert is_text_generation_model(model_cls)

    # All vLLM models should be convertible to a pooling model
    assert is_pooling_model(as_seq_cls_model(model_cls))
    assert is_pooling_model(as_embedding_model(model_cls))

    if model_arch in _MULTIMODAL_MODELS:
        assert supports_multimodal(model_cls)


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch,is_mm,init_cuda,score_type",
    [
        ("LlamaForCausalLM", False, False, "bi-encoder"),
        ("LlavaForConditionalGeneration", True, True, "bi-encoder"),
        ("BertForSequenceClassification", False, False, "cross-encoder"),
        ("RobertaForSequenceClassification", False, False, "cross-encoder"),
        ("XLMRobertaForSequenceClassification", False, False, "cross-encoder"),
        ("GteNewModel", False, False, "bi-encoder"),
        ("GteNewForSequenceClassification", False, False, "cross-encoder"),
        ("HF_ColBERT", False, False, "late-interaction"),
    ],
)
def test_registry_model_property(model_arch, is_mm, init_cuda, score_type):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_multimodal is is_mm
    assert model_info.score_type == score_type

    if init_cuda and current_platform.is_cuda_alike():
        assert not torch.cuda.is_initialized()

        ModelRegistry._try_load_model_cls(model_arch)
        if not torch.cuda.is_initialized():
            warnings.warn(
                "This model no longer initializes CUDA on import. "
                "Please test using a different one.",
                stacklevel=2,
            )


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch,is_pp,init_cuda",
    [
        # TODO(woosuk): Re-enable this once the MLP Speculator is supported
        # in V1.
        # ("MLPSpeculatorPreTrainedModel", False, False),
        ("DeepseekV2ForCausalLM", True, False),
        ("Qwen2VLForConditionalGeneration", True, True),
    ],
)
def test_registry_is_pp(model_arch, is_pp, init_cuda):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None

    assert model_info.supports_pp is is_pp

    if init_cuda and current_platform.is_cuda_alike():
        assert not torch.cuda.is_initialized()

        ModelRegistry._try_load_model_cls(model_arch)
        if not torch.cuda.is_initialized():
            warnings.warn(
                "This model no longer initializes CUDA on import. "
                "Please test using a different one.",
                stacklevel=2,
            )


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_arch,supported",
    [
        # ReplaySSM is opt-in per model; only Nemotron-H sets the flag today.
        ("NemotronHForCausalLM", True),
        ("Mamba2ForCausalLM", False),
        ("Zamba2ForCausalLM", False),
    ],
)
def test_registry_supports_replayssm(model_arch, supported):
    model_info = ModelRegistry._try_inspect_model_cls(model_arch)
    assert model_info is not None
    assert model_info.supports_replayssm is supported


def test_lazy_modelinfo_package_hash_includes_submodules(tmp_path):
    package_dir = tmp_path / "model_package"
    package_dir.mkdir()
    init_file = package_dir / "__init__.py"
    init_file.write_text("from .model import Model\n", encoding="utf-8")
    model_file = package_dir / "model.py"
    model_file.write_text("class Model: pass\n", encoding="utf-8")

    first_hash = _LazyRegisteredModel._get_modelinfo_module_hash(init_file)

    model_file.write_text("class Model:\n    supports_pp = True\n", encoding="utf-8")
    second_hash = _LazyRegisteredModel._get_modelinfo_module_hash(init_file)

    assert first_hash != second_hash


def test_hf_registry_coverage():
    untested_archs = (
        ModelRegistry.get_supported_archs() - HF_EXAMPLE_MODELS.get_supported_archs()
    )

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}"
    )


@pytest.fixture
def _fresh_warning_cache():
    """`logger.warning_once` memoises on (msg, *args), so a warning emitted earlier in
    the session would make these assertions depend on test order."""
    _print_warning_once.cache_clear()
    yield
    _print_warning_once.cache_clear()


def test_oot_architecture_warns_and_names_the_plugin(caplog, _fresh_warning_cache):
    """The plugin pointer must reach the user under the default `--model-impl auto`.

    It is only ever read in `_raise_for_unsupported`, which runs after the Transformers
    fallback has had its turn. For any architecture Transformers implements the fallback
    succeeds, so without this warning the operator silently gets a different
    implementation and the plugin URL is never shown.
    """
    arch = next(iter(_OOT_SUPPORTED_MODELS))
    expected = _OOT_SUPPORTED_MODELS[arch]
    model_config = SimpleNamespace(model_impl="auto")

    with caplog.at_level(logging.WARNING, logger="vllm.model_executor.models.registry"):
        ModelRegistry._warn_for_evicted_architectures([arch], model_config)

    assert arch in caplog.text
    assert expected in caplog.text


def test_previously_supported_architecture_warns_with_its_last_version(
    caplog, _fresh_warning_cache
):
    arch = next(iter(_PREVIOUSLY_SUPPORTED_MODELS))
    version = _PREVIOUSLY_SUPPORTED_MODELS[arch]
    model_config = SimpleNamespace(model_impl="auto")

    with caplog.at_level(logging.WARNING, logger="vllm.model_executor.models.registry"):
        ModelRegistry._warn_for_evicted_architectures([arch], model_config)

    assert arch in caplog.text
    assert version in caplog.text


def test_no_warning_once_the_plugin_is_installed(caplog, _fresh_warning_cache):
    """The case a plugin exists to produce: registered wins, and says nothing."""
    arch = next(iter(_OOT_SUPPORTED_MODELS))
    model_config = SimpleNamespace(model_impl="auto")

    ModelRegistry.register_model(
        arch, "vllm.model_executor.models.llama:LlamaForCausalLM"
    )
    try:
        with caplog.at_level(
            logging.WARNING, logger="vllm.model_executor.models.registry"
        ):
            ModelRegistry._warn_for_evicted_architectures([arch], model_config)
    finally:
        ModelRegistry.models.pop(arch, None)

    assert arch not in caplog.text


@pytest.mark.parametrize("model_impl", ["transformers", "terratorch"])
def test_no_warning_when_the_user_chose_the_implementation(
    model_impl, caplog, _fresh_warning_cache
):
    """Only `auto` silently substitutes. If the operator named an implementation, they
    are not being surprised and do not need telling."""
    arch = next(iter(_OOT_SUPPORTED_MODELS))
    model_config = SimpleNamespace(model_impl=model_impl)

    with caplog.at_level(logging.WARNING, logger="vllm.model_executor.models.registry"):
        ModelRegistry._warn_for_evicted_architectures([arch], model_config)

    assert arch not in caplog.text


def test_eviction_tables_are_well_formed():
    """Neither table had any test coverage before this file. Both are hand-maintained
    and only ever read on an error path, so a typo in one is invisible until a user
    hits it."""
    for arch, url in _OOT_SUPPORTED_MODELS.items():
        assert arch.isidentifier(), arch
        assert url.startswith("https://"), (arch, url)

    for arch, version in _PREVIOUSLY_SUPPORTED_MODELS.items():
        assert arch.isidentifier(), arch
        parts = version.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts), (arch, version)

    overlap = set(_OOT_SUPPORTED_MODELS) & set(_PREVIOUSLY_SUPPORTED_MODELS)
    assert not overlap, (
        f"{overlap} are in both tables; _raise_for_unsupported checks "
        "_PREVIOUSLY_SUPPORTED_MODELS first, so the plugin pointer would never show"
    )

    live = ModelRegistry.get_supported_archs()
    for table_name, table in (
        ("_OOT_SUPPORTED_MODELS", _OOT_SUPPORTED_MODELS),
        ("_PREVIOUSLY_SUPPORTED_MODELS", _PREVIOUSLY_SUPPORTED_MODELS),
    ):
        clash = set(table) & set(live)
        assert not clash, (
            f"{clash} are both registered in-tree and listed in {table_name}; "
            "the eviction message is unreachable for them"
        )
