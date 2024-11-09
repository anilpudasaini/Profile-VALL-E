import pandas as pd
from pathlib import Path
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, fix_manifests, validate_recordings_and_supervisions
import logging
from typing import Dict, Union

def prepare_custom_dataset(
    metadata_csv: Path,
    output_dir: Path,
    target_sample_rate: int = 24000
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare RecordingSet and SupervisionSet for a custom dataset using a CSV with absolute paths.

    :param metadata_csv: Path to the CSV file with metadata (e.g., path, sentence, age_group, etc.).
    :param output_dir: Directory to save the generated manifests.
    :param target_sample_rate: Desired sample rate for all recordings.
    :return: Dictionary containing RecordingSet and SupervisionSet.
    """
    # Load the metadata from the CSV file
    metadata_df = pd.read_csv(metadata_csv)

    recordings = []
    supervisions = []
    recording_ids = set()

    # Iterate through each row in the metadata CSV
    for index, row in metadata_df.iterrows():
        audio_path = Path(row["path"])  # Using the absolute path directly
        recording_id = audio_path.stem  # Base ID from filename
        supervision_id = f"{recording_id}_{row['style_id']}_{index}"  # Unique ID using index  # Unique ID for each style_id

        # Check if audio file exists
        if not audio_path.is_file():
            logging.warning(f"Missing audio file: {audio_path}")
            continue

        # Only add the recording once, even if it has multiple supervisions with different style_id
        if recording_id not in recording_ids:
            recording = Recording.from_file(audio_path, recording_id=recording_id)
            if recording.sampling_rate != target_sample_rate:
                recording = recording.resample(target_sample_rate)
            recordings.append(recording)
            recording_ids.add(recording_id)

        # Create SupervisionSegment with custom metadata and trimmed duration
        supervision = SupervisionSegment(
            id=supervision_id,
            recording_id=recording_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            text=row["sentence"],  # Transcript text
            language="English",
            speaker=row["style_id"],  # style_id to represent speaker profile
            custom={
                "age_group": row["age_group"],
                "gender": row["gender"],
                "accent": row["simplified_accents"],
                "orig_text": row["sentence"],
                "style_id": row["style_id"]
            }
        )
        supervisions.append(supervision)

    # Create RecordingSet and SupervisionSet
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    # Fix and validate manifests
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    # Save manifests to JSONL files
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_set.to_file(output_dir / "custom_recordings.jsonl.gz")
    supervision_set.to_file(output_dir / "custom_supervisions.jsonl.gz")

    return {"recordings": recording_set, "supervisions": supervision_set}


prepare_custom_dataset(
    metadata_csv=Path("/home/anil/cv-corpus-17.0-2024-03-15/en/dataset/added_style_wav.csv"),
    output_dir=Path("/home/anil/cv-corpus-17.0-2024-03-15/en/manifests")
)
