"""Bounded real-model attention-trace smoke test for the FIER→WindowQuant study.

This captures attention tensors from SmolVLM-256M on two fixed public images.
It is a trace/contract diagnostic, not a task-quality benchmark or novelty test.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from io import BytesIO
from pathlib import Path
import numpy as np
import requests
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
MODEL_REVISION = "d039ab0d4b9311b617e8ffec1d06c8464364aeb3"
IMAGES = {
    "candy": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
    "statue": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
}
PROMPTS = {
    "candy": "What animal is on the candy?",
    "statue": "What landmark is shown in this image?",
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def finite(x: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(x).all().item()):
        raise AssertionError(f"non-finite {name}")

def image_positions(ids: torch.Tensor) -> list[int]:
    # Idefics3 expands an image marker to image placeholder IDs 49153..49278.
    return [i for i, v in enumerate(ids.tolist()) if 49153 <= int(v) <= 49278]

def mean_attention(attn, query_indices, key_indices):
    # Return one scalar per layer: mean head/query/key mass into visual keys.
    values = []
    for layer in attn:
        if layer is None:
            raise AssertionError("missing eager attention tensor")
        finite(layer, "attention")
        if not query_indices or not key_indices:
            values.append(np.nan); continue
        x = layer[0, :, query_indices, :][:, :, key_indices]
        values.append(float(x.mean().detach().cpu()))
    return np.asarray(values, dtype=np.float32)

def run_case(name, model, processor, image_path: Path, device: str, output_dir: Path):
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[name]}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ids = inputs["input_ids"][0].detach().cpu()
    visual = image_positions(ids)
    if not visual:
        raise AssertionError("no visual placeholder positions")
    # Only text positions after the visual span can attend to the visual span
    # under the causal mask. This is a structural trace, not a relevance label.
    text_after = [i for i in range(ids.numel()) if i > max(visual) and int(ids[i]) not in visual]
    t0 = time.time()
    with torch.inference_mode():
        first = model(**inputs, output_attentions=True, use_cache=True, return_dict=True)
    prefill_seconds = time.time() - t0
    prefill_visual = mean_attention(first.attentions, text_after, visual)
    next_ids = []
    decode_visual = []
    past = first.past_key_values
    mask = inputs["attention_mask"]
    current = first.logits[:, -1:].argmax(dim=-1)
    for step in range(4):
        next_ids.append(int(current.item()))
        mask = torch.cat([mask, torch.ones((1, 1), device=device, dtype=mask.dtype)], dim=1)
        with torch.inference_mode():
            out = model(input_ids=current, attention_mask=mask, past_key_values=past,
                        output_attentions=True, use_cache=True, return_dict=True)
        past = out.past_key_values
        # Decode query has length one; key range includes the full cached prefix.
        row = []
        for layer in out.attentions:
            finite(layer, "decode attention")
            if layer.shape[-2:] != (1, mask.shape[-1]):
                raise AssertionError("unexpected decode attention shape")
            row.append(float(layer[0, :, 0, :].mean().detach().cpu()))
        decode_visual.append(row)
        current = out.logits[:, -1:].argmax(dim=-1)
    finite(first.logits, "prefill logits")
    attention_sums = []
    for layer in first.attentions:
        sums = layer.sum(dim=-1).detach().cpu().numpy()
        attention_sums.append(float(np.max(np.abs(sums - 1.0))))
    trace = {
        "input_ids": ids.numpy(),
        "visual_positions": np.asarray(visual, dtype=np.int32),
        "prefill_visual_attention_mean": prefill_visual,
        "decode_attention_mean": np.asarray(decode_visual, dtype=np.float32),
        "generated_ids": np.asarray(next_ids, dtype=np.int64),
    }
    npz = output_dir / f"{name}.npz"
    np.savez_compressed(npz, **trace)
    return {
        "image": name, "prompt": PROMPTS[name], "input_tokens": int(ids.numel()),
        "visual_tokens": len(visual), "layers": len(first.attentions),
        "heads": int(first.attentions[0].shape[1]), "prefill_seconds": prefill_seconds,
        "generated_text": processor.decode(next_ids, skip_special_tokens=False),
        "generated_ids": next_ids, "prefill_attention_max_row_sum_error": max(attention_sums),
        "decode_steps": 4, "npz": npz.name, "npz_sha256": sha(npz),
        "device": device,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("../../smolvlm-model-d039ab0"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    a = p.parse_args(); out = a.output
    if out.exists() and any(out.iterdir()): raise FileExistsError("preserve completed output")
    out.mkdir(parents=True, exist_ok=True); (out / "images").mkdir()
    receipts = []
    for name, url in IMAGES.items():
        path = out / "images" / f"{name}.jpg"
        response = requests.get(url, timeout=60); response.raise_for_status()
        if len(response.content) > 20_000_000: raise ValueError("image exceeds 20MB")
        path.write_bytes(response.content)
        Image.open(path).convert("RGB").resize((512, 512)).save(path, format="JPEG", quality=95)
        receipts.append({"name": name, "url": url, "sha256": sha(path), "bytes": path.stat().st_size})
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForVision2Seq.from_pretrained(a.model, local_files_only=True, dtype=torch.float32).eval()
    model.set_attn_implementation("eager")
    device = "mps" if a.device == "auto" and torch.backends.mps.is_available() else ("cpu" if a.device == "auto" else a.device)
    try: model.to(device)
    except Exception as exc:
        if a.device == "auto": device = "cpu"; model.to(device)
        else: raise
    rows = [run_case(name, model, processor, out / "images" / f"{name}.jpg", device, out) for name in IMAGES]
    manifest = {
        "schema": "smolvlm-attention-trace.v1", "status": "completed_trace_smoke_only",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_path": str(a.model), "device": device, "torch": torch.__version__,
        "python": platform.python_version(), "transformers": __import__("transformers").__version__,
        "processor_image_splitting": False, "images": receipts, "cases": rows,
        "scope": "Real attention traces only; no task-quality, usefulness, latency, or novelty claim.",
        "constraints_checked": ["finite attention/logits", "attention row normalization", "visual placeholder positions", "four greedy decode steps"],
        "source": "Hugging Face model card and documentation-image URLs; source bytes are hashed above.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "device": device, "cases": len(rows), "output": str(out)}, indent=2))

if __name__ == "__main__": main()
