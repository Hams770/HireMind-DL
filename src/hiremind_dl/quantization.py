"""M2: post-training quantization of M1 (no retraining). LayerNorm stays FP32."""
import torch
from torch.ao.quantization import (
    default_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    quantize_dynamic,
)


def quantize(model, variant="both"):
    """variant: "linear" (dynamic INT8 Linear), "embeddings" (8-bit weight-only), or "both" (= M2)."""
    specs = {}
    if variant in ["linear", "both"]:
        specs[torch.nn.Linear] = default_dynamic_qconfig
    if variant in ["embeddings", "both"]:
        specs[torch.nn.Embedding] = float_qparams_weight_only_qconfig
    if not specs:
        raise ValueError(variant)
    engines = torch.backends.quantized.supported_engines
    engine = "x86" if "x86" in engines else "fbgemm"
    if engine not in engines:
        raise RuntimeError("Quantization needs the x86/fbgemm CPU backend.")
    torch.backends.quantized.engine = engine
    q = quantize_dynamic(model.cpu().eval(), qconfig_spec=specs, inplace=False)
    if variant in ["embeddings", "both"]:
        assert q.bert.embeddings.word_embeddings.weight().dtype == torch.quint8
    return q
