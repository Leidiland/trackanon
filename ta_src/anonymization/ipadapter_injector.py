"""Splices the IP-Adapter-FaceID-Plus-v2 chain into KSampler model edges at runtime."""
from __future__ import annotations

import copy
from dataclasses import dataclass

_PRESET = "FACEID PLUS V2"

# Stable IDs keep re-injection idempotent (find-and-refresh, not append).
_NID_LOADER = "_ipa_loader"
_NID_IMAGE = "_ipa_image"
_NID_IPA = "_ipa_face"  # legacy single-chain id
_NID_IPA_PREFIX = "_ipa_face_"  # multi-chain: _ipa_face_0, _ipa_face_1, ...


@dataclass(frozen=True)
class ApplyConfig:
    """One IPAdapterFaceID apply node's wiring for a multi-chain inject."""
    target_ksampler_id: str
    face_strength: float
    body_strength: float
    weight_type: str = "linear"
    start_at: float = 0.0
    end_at: float = 1.0


def inject(
    workflow: dict,
    reference_crop_filename: str,
    *,
    lora_strength: float,
    lora_filename: str,  # noqa: ARG001 — kept for caller API; loader resolves the LoRA from preset
    # Legacy single-chain kwargs (preserved verbatim from ADR-0004):
    face_strength: float | None = None,
    body_strength: float | None = None,
    weight_type: str = "style transfer",
    start_at: float = 0.2,
    end_at: float = 1.0,
    # Multi-chain (ADR-0019 face polish):
    apply_configs: list[ApplyConfig] | None = None,
) -> dict:
    """Inject the FaceID-Plus-v2 chain into a workflow dict (deep-copied, idempotent).

    Two modes:
      * Legacy single-chain (apply_configs=None): one IPAdapterFaceID apply
        node feeding the workflow's only KSampler. Behavior is byte-equivalent
        to the original ADR-0004 implementation.
      * Multi-chain (apply_configs provided): one shared IPAdapterUnifiedLoaderFaceID,
        N IPAdapterFaceID apply nodes (stable IDs _ipa_face_0 ... _ipa_face_{N-1}),
        each feeding its declared target KSampler's `model` input. lora_strength
        stays single-valued at the loader (ADR-0019).
    """
    if apply_configs is not None:
        return _inject_multi(
            workflow,
            reference_crop_filename,
            lora_strength=lora_strength,
            apply_configs=apply_configs,
        )

    if face_strength is None or body_strength is None:
        raise ValueError(
            "inject: legacy single-chain call requires face_strength and body_strength"
        )

    wf = copy.deepcopy(workflow)
    ksampler_id = _find_node(wf, "KSampler")
    if ksampler_id is None:
        raise ValueError(
            "inject: no KSampler node found in workflow — cannot wire IP-Adapter conditioning"
        )

    already_injected = _NID_IPA in wf
    original_model = (
        wf[_NID_LOADER]["inputs"]["model"]
        if already_injected
        else wf[ksampler_id]["inputs"]["model"]
    )

    wf[_NID_LOADER] = _loader_node(original_model, lora_strength)
    wf[_NID_IMAGE] = _image_node(reference_crop_filename)
    wf[_NID_IPA] = _apply_node(
        face_strength=face_strength,
        body_strength=body_strength,
        weight_type=weight_type,
        start_at=start_at,
        end_at=end_at,
    )
    wf[ksampler_id]["inputs"]["model"] = [_NID_IPA, 0]
    return wf


def _inject_multi(
    workflow: dict,
    reference_crop_filename: str,
    *,
    lora_strength: float,
    apply_configs: list[ApplyConfig],
) -> dict:
    wf = copy.deepcopy(workflow)
    for cfg in apply_configs:
        if cfg.target_ksampler_id not in wf:
            raise ValueError(
                f"inject: target_ksampler_id={cfg.target_ksampler_id!r} not present in workflow"
            )
        if wf[cfg.target_ksampler_id].get("class_type") != "KSampler":
            raise ValueError(
                f"inject: target_ksampler_id={cfg.target_ksampler_id!r} is not a KSampler node"
            )

    # Resolve the loader's upstream model edge: prefer the first target's
    # current pre-injection model (idempotency: re-running picks up the
    # loader's stored model, not the apply-node loopback).
    first_target = apply_configs[0].target_ksampler_id
    if _NID_LOADER in wf:
        original_model = wf[_NID_LOADER]["inputs"]["model"]
    else:
        original_model = wf[first_target]["inputs"]["model"]

    wf[_NID_LOADER] = _loader_node(original_model, lora_strength)
    wf[_NID_IMAGE] = _image_node(reference_crop_filename)

    for idx, cfg in enumerate(apply_configs):
        apply_id = f"{_NID_IPA_PREFIX}{idx}"
        wf[apply_id] = _apply_node(
            face_strength=cfg.face_strength,
            body_strength=cfg.body_strength,
            weight_type=cfg.weight_type,
            start_at=cfg.start_at,
            end_at=cfg.end_at,
        )
        wf[cfg.target_ksampler_id]["inputs"]["model"] = [apply_id, 0]

    return wf


def _loader_node(upstream_model, lora_strength: float) -> dict:
    return {
        "class_type": "IPAdapterUnifiedLoaderFaceID",
        "inputs": {
            "model": upstream_model,
            "preset": _PRESET,
            "lora_strength": float(lora_strength),
            "provider": "CPU",
        },
    }


def _image_node(filename: str) -> dict:
    return {
        "class_type": "LoadImage",
        "inputs": {"image": filename},
    }


def _apply_node(
    *,
    face_strength: float,
    body_strength: float,
    weight_type: str,
    start_at: float,
    end_at: float,
) -> dict:
    return {
        "class_type": "IPAdapterFaceID",
        "inputs": {
            "model": [_NID_LOADER, 0],
            "ipadapter": [_NID_LOADER, 1],
            "image": [_NID_IMAGE, 0],
            "weight": float(face_strength),
            "weight_faceidv2": float(body_strength),
            "weight_type": str(weight_type),
            "combine_embeds": "concat",
            "start_at": float(start_at),
            "end_at": float(end_at),
            "embeds_scaling": "V only",
        },
    }


def _find_node(workflow: dict, class_type: str) -> str | None:
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return node_id
    return None
