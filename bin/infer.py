#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import get_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-prompts",
        type=str,
        default="",
        help="Text prompts separated by |.",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts separated by |, aligned with --text-prompts.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="To get up and running quickly just follow the steps below.",
        help="Text to synthesize.",
    )

    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="Text extraction method: espeak, pypinyin, etc.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved model checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Directory to store output files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="AR Decoder top-k sampling (if > 0).",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="AR Decoder temperature for top-k sampling.",
    )

    parser.add_argument(
        "--profile-description",
        type=str,
        default="An adult male with British accent",
        help="Text description of speaker profile (e.g. 'A young male with British accent')",
    )

    return parser.parse_args()


def load_model(checkpoint, device):
    checkpoint = torch.load(checkpoint, map_location=device)

    
    args = AttributeDict(checkpoint)
    model = get_model(args)
 
    # check to see if profile encoder keys are present 
    assert hasattr(model, 'profile_encoder'), "Model missing profile encoder!"
    
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    text_tokens = args.text_tokens

    return model, text_tokens


@torch.no_grad()
def main():
    args = get_args()
    text_tokenizer = TextTokenizer(backend=args.text_extractor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, text_tokens = load_model(args.checkpoint, device)
    text_collater = get_text_token_collater(text_tokens)

    audio_tokenizer = AudioTokenizer()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    text_prompts = " ".join(args.text_prompts.split("|"))

    # Process profile description
    profile_description = args.profile_description.strip()
    if not profile_description:
        raise ValueError("Profile description cannot be empty!!!")

    for n, text in enumerate(args.text.split("|")):
        logging.info(f"Synthesizing text: {text}")

        # Tokenize input text
        text_tokens, text_tokens_lens = text_collater(
            [tokenize_text(text_tokenizer, text=f"{text_prompts} {text}".strip())]
        )

        # Handle audio prompts (optional)
        audio_prompts = None
        if args.audio_prompts:
            audio_prompts = [
                tokenize_audio(audio_tokenizer, audio_file)[0][0]
                for audio_file in args.audio_prompts.split("|")
            ]
            audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1).to(device)

        # Perform inference
        with torch.inference_mode():
            encoded_frames = model.inference(
                x=text_tokens.to(device),
                x_lens=text_tokens_lens.to(device),
                y=audio_prompts,
                profile_prompt=[profile_description],  # Pass profile description (Note: singular not plural)
                top_k=args.top_k,
                temperature=args.temperature,
            )

        # Create filename-safe version of profile description
        safe_description = "".join(
            [c if c.isalnum() else "_" for c in profile_description]
        )[:50]
        
        # Decode and save audio
        samples = audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])
        torchaudio.save(
            f"{args.output_dir}/{n}_profile_{safe_description}.wav",
            samples[0].cpu(),
            24000
        )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
