"""Download pretrained model weights once into ./models - after this everything runs offline.

    python download_models.py                  # default: the formula and English readers
    python download_models.py trocr-base-handwritten   # optional: the bigger English reader (EXAM_GRADER_ENGLISH=base)
    python download_models.py dictalm-3.0-1.7b-base    # optional: Hebrew read in context (recog/hebrew_context.py)
    python download_models.py qwen3-vl-2b      # optional VLMs for experiments (see recog/hebrew_vlm.py)
    python download_models.py --list

The Hebrew line recognizer (models/hebrew-crnn) is not downloaded - it is trained locally
by train_hebrew.py. Re-running is safe: files already present are skipped.
"""
import os
import sys

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    # name in ./models         Hugging Face repo                                   size
    "pix2text-mfr":            ("breezedeus/pix2text-mfr",                          "0.1 GB  formulas (ONNX)"),
    "trocr-small-handwritten": ("Xenova/trocr-small-handwritten",                   "0.25 GB English handwriting (ONNX export of microsoft/trocr-small-handwritten)"),
    "trocr-base-handwritten":  ("Xenova/trocr-base-handwritten",                    "0.6 GB  better English handwriting, ~5x slower (EXAM_GRADER_ENGLISH=base)"),
    "qwen3-vl-2b":             ("Qwen/Qwen3-VL-2B-Instruct",                        "4.3 GB  general VLM (zero-shot Hebrew handwriting: poor)"),
    "qwen3-vl-8b":             ("Qwen/Qwen3-VL-8B-Instruct",                        "17 GB   bigger VLM"),
    "kohelet-hebrew-htr-lora": ("kohelet-splendour/qwen3-vl-hebrew-htr-lines-finetuned",
                                "0.2 GB  community Hebrew-HTR LoRA for qwen3-vl-8b (untested)"),
    "dictalm-3.0-1.7b-base":   ("dicta-il/DictaLM-3.0-1.7B-Base",
                                "3.4 GB  Hebrew language model: Hebrew text read in context (EXAM_GRADER_HEBREW_CONTEXT=1)"),
    "rtdetr-v2-r18":           ("PekingU/rtdetr_v2_r18vd",
                                "0.08 GB base of the circuit detector (RT-DETRv2, Apache-2.0; tools/train_circuit_detector.py)"),
    "uni-mumer-qwen3vl-2b":    ("phxember/Uni-MuMER-Qwen3-VL-2B",
                                "4.3 GB  handwritten-formula reader (Qwen3-VL-2B fine-tuned, CROHME 2014 83 % exact; GPU advised)"),
}
DEFAULT = ["pix2text-mfr", "trocr-small-handwritten"]
# repos whose ONNX files sit in an onnx/ subfolder: only these files are fetched, then moved up
ONNX_ONLY = {"trocr-small-handwritten": ["config.json", "generation_config.json", "preprocessor_config.json",
                                         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                         "onnx/encoder_model.onnx", "onnx/decoder_model.onnx"],
             # full-precision encoder (its 8-bit export reads IAM at 41 % CER - broken) + the 8-bit decoder
             "trocr-base-handwritten": ["config.json", "generation_config.json", "preprocessor_config.json",
                                        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                        "onnx/encoder_model.onnx", "onnx/decoder_model_quantized.onnx"]}
RENAME = {"decoder_model_quantized.onnx": "decoder_model.onnx"}

if "--list" in sys.argv:
    for k, (repo, size) in MODELS.items():
        print(f"{k:25} {repo:55} {size}")
    sys.exit(0)

names = [a for a in sys.argv[1:] if not a.startswith("-")] or DEFAULT
for name in names:
    repo, size = MODELS[name]
    target = os.path.join(HERE, "models", name)
    print(f"-> {name}  ({repo}, {size})  into {target}", flush=True)
    if name in ONNX_ONLY:
        if all(os.path.isfile(os.path.join(target, f)) for f in ("encoder_model.onnx", "decoder_model.onnx", "tokenizer.json")):
            print("   already there")
            continue
        snapshot_download(repo, local_dir=target, allow_patterns=ONNX_ONLY[name])
        for f in ("encoder_model.onnx", "decoder_model.onnx", "decoder_model_quantized.onnx"):
            src, dst = os.path.join(target, "onnx", f), os.path.join(target, RENAME.get(f, f))
            if os.path.isfile(src) and not os.path.isfile(dst):
                os.replace(src, dst)
        try:
            os.rmdir(os.path.join(target, "onnx"))           # now empty
        except OSError:
            pass
        continue
    snapshot_download(repo, local_dir=target,
                      ignore_patterns=["*.md", ".gitattributes", "*.pt", "*.bin"] if name != "pix2text-mfr" else [".gitattributes"])
print("done")
