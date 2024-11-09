#!/usr/bin/env python3
import argparse
import logging
import os
import json
from pathlib import Path
import pandas as pd
import torch
import torch.multiprocessing
from tqdm.auto import tqdm
from lhotse import Recording, SupervisionSegment, CutSet, MonoCut, NumpyHdf5Writer

from valle.data import (
    AudioTokenConfig,
    AudioTokenExtractor,
    TextTokenizer,
    tokenize_text,
)
from valle.utils import SymbolTable

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized files",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to the metadata CSV file containing style_id and other info",
    )
    return parser.parse_args()

def process_and_tokenize(metadata_df, output_dir):
    # Initialize the text tokenizer and audio extractor
    text_tokenizer = TextTokenizer(backend="espeak")  # Change backend if needed
    audio_extractor = AudioTokenExtractor(AudioTokenConfig())
    
    unique_symbols = set()
    cuts = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        file_path = row['path']
        sentence = row['sentence']
        style_id = row['style_id']
        
        # Create a Recording from the audio file
        recording = Recording.from_file(file_path)
        
        # Create a SupervisionSegment with text and style_id
        supervision = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            text=sentence,
            custom={"style_id": style_id}
        )
        
        # Create a MonoCut with features, recording, and supervision information
        cut = MonoCut(
            id=f"{recording.id}-0",
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
            supervisions=[supervision],
            features=None,  # Placeholder until features are computed
            custom={"dataloading_info": {"rank": 0, "world_size": 1, "worker_id": None}}
        )

        # Compute and store audio features, attaching them to the cut
        if audio_extractor:
            cut = cut.compute_and_store_features(
                extractor=audio_extractor,
                storage_path=f"{output_dir}/audio_features",
                storage_type=NumpyHdf5Writer,
            )
        
        # Tokenize text and add phonemes to the `custom` field in supervision
        if text_tokenizer:
            phonemes = tokenize_text(text_tokenizer, text=sentence)
            supervision.custom["tokens"] = {"text": phonemes}
            unique_symbols.update(phonemes)

        # Convert the cut to a dictionary for JSONL storage
        cuts.append(cut.to_dict())

    # Save each cut as a JSONL file line
    with open(f"{output_dir}/cuts_all.jsonl", "w") as f:
        for cut_dict in cuts:
            f.write(json.dumps(cut_dict) + "\n")

    # Save unique phonemes for text
    with open(f"{output_dir}/unique_text_tokens.txt", "w") as f:
        f.write("\n".join(sorted(unique_symbols)))

def main():
    args = get_args()

    # Load metadata with style_id
    metadata_df = pd.read_csv(args.metadata_path)
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process and tokenize the entire CSV dataset
    process_and_tokenize(metadata_df, output_dir=args.output_dir)

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
