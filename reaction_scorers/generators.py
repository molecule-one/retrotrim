"""Public RootAligned generator used by the supported study profile."""

import os
from pathlib import Path
from unittest.mock import patch

DEFAULT_GENERATOR = "rootaligned"
DEFAULT_NUM_RESULTS = 10
DEFAULT_NUM_AUGMENTATIONS = 1


def load_public_generator(device="cpu", cache_dir=None, *, generator=DEFAULT_GENERATOR,
                          num_augmentations=DEFAULT_NUM_AUGMENTATIONS,
                          num_results=DEFAULT_NUM_RESULTS, canonicalization_workers=2):
    if generator != "rootaligned":
        raise ValueError("Only RootAligned is supported; other generators require independent calibration and validation")
    if num_results < 1 or num_augmentations < 1 or canonicalization_workers < 1:
        raise ValueError("Result, augmentation, and worker counts must be positive")
    if cache_dir is not None:
        os.environ["SYNTHESEUS_CACHE_DIR"] = str(Path(cache_dir).resolve())
    return _load_root_aligned(device, num_augmentations, num_results, canonicalization_workers)


def _load_root_aligned(device, num_augmentations, num_results, workers):
    import torch
    from syntheseus.reaction_prediction.inference.root_aligned import RootAlignedModel
    from syntheseus.reaction_prediction.utils.downloading import get_default_model_dir_from_cache
    from syntheseus.reaction_prediction.utils.inference import get_unique_file_in_dir

    model_dir = get_default_model_dir_from_cache("RootAligned", is_forward=False)
    if model_dir is None:
        raise RuntimeError("Syntheseus has no public RootAligned checkpoint configured")
    checkpoint = Path(get_unique_file_in_dir(model_dir, pattern="*.pt")).resolve()
    original_load = torch.load

    def load_checkpoint(path, *args, **kwargs):
        # The public legacy checkpoint includes torchtext vocabulary/Namespace
        # objects. Relax Torch's tensor-only loading for this exact file only.
        if isinstance(path, (str, Path)) and Path(path).resolve() == checkpoint:
            kwargs["weights_only"] = False
        return original_load(path, *args, **kwargs)

    class RootAlignedWithWorkers(RootAlignedModel):
        def _get_reactions(self, inputs, num_results, **kwargs):
            # Upstream otherwise creates one process per visible CPU for each
            # prediction. Scope the override to this call, not the whole session.
            with patch("multiprocessing.cpu_count", return_value=workers):
                return super()._get_reactions(inputs, num_results, **kwargs)

    with patch("torch.load", load_checkpoint):
        model = RootAlignedWithWorkers(model_dir=model_dir, device=device, use_cache=True,
                                       num_augmentations=num_augmentations,
                                       default_num_results=num_results)
    model.public_generator_name = "RootAligned"
    model.public_generator_settings = {"num_augmentations": num_augmentations,
                                       "num_results": num_results,
                                       "canonicalization_workers": workers}
    return model
