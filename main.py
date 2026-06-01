"""Fact-Check Agent — console runner (ReAct agent).

Usage:
    python main.py
    python main.py --claim "Biden said X" --image path/to/img.jpg --lang vi
"""

import argparse
import asyncio
import os
import uuid

import torch

# ── Model import ─────────────────────────────────────────────────────
import similarity.model.longclip as longclip

# ── Agent imports ────────────────────────────────────────────────────
import agent.tools as tools_module
from agent.react_agent import run_react_agent


# ═══════════════════════════════════════════════════════════════════════
# Model loader
# ═══════════════════════════════════════════════════════════════════════
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading LongCLIP on {device.upper()}...")
    model, preprocess = longclip.load(
        "./similarity/checkpoints/longclip-B.pt", device=device
    )
    model.eval()
    # Inject vào tool registry
    tools_module.set_models(model, preprocess, device)
    print("[+] Model loaded.\n")
    return device


# ═══════════════════════════════════════════════════════════════════════
# Interactive console prompt
# ═══════════════════════════════════════════════════════════════════════
def prompt_inputs():
    """Interactive prompt khi không có CLI args."""
    print("\n=== Fact-Check Agent ===\n")

    claim = input("Claim text (để trống nếu chỉ có ảnh): ").strip()

    images = []
    while True:
        img = input("Image path (để trống khi xong): ").strip()
        if not img:
            break
        if os.path.isfile(img):
            images.append(img)
        else:
            print(f"  ⚠ File không tồn tại: {img}")

    if not claim and not images:
        print("❌ Cần cung cấp ít nhất một claim hoặc một ảnh.")
        raise SystemExit(1)

    lang = input("Ngôn ngữ [en/vi] (mặc định: en): ").strip() or "en"
    return claim, images, lang


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fact-Check Agent (console)")
    parser.add_argument("--claim", type=str, default="",  help="Claim cần kiểm chứng")
    parser.add_argument("--image", type=str, action="append", default=[], help="Đường dẫn ảnh (có thể lặp nhiều lần)")
    parser.add_argument("--lang",  type=str, default="en", help="Ngôn ngữ: en hoặc vi")
    args = parser.parse_args()

    # Nếu không truyền arg → hỏi interactive
    if not args.claim and not args.image:
        claim_text, image_paths, language = prompt_inputs()
    else:
        claim_text  = args.claim
        image_paths = args.image
        language    = args.lang

    # Load model
    load_model()

    # Chạy ReAct agent
    print("=" * 60)
    print("Bắt đầu kiểm chứng...")
    print("=" * 60 + "\n")

    final_report = asyncio.run(run_react_agent(claim_text, image_paths, language))

    print("\n" + "=" * 60)
    print("BÁO CÁO CUỐI CÙNG")
    print("=" * 60)
    print(final_report)
