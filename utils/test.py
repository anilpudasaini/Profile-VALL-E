# import torchaudio
# print(torchaudio.get_audio_backend())
# import torchaudio
# torchaudio.set_audio_backend("ffmpeg")
# print(torchaudio.get_audio_backend())  # Should output 'ffmpeg'
import pandas as pd

# Load the CSV file
csv_path = "/home/anil/cv-corpus-17.0-2024-03-15/en/dataset/added_style.csv"
df = pd.read_csv(csv_path)

# Check for any remaining .mp3 paths
mp3_files = df[df['path'].str.endswith(".mp3")]

if not mp3_files.empty:
    print("Some .mp3 files are still present in the CSV:")
    print(mp3_files['path'])
else:
    print("All paths have been updated to .wav")
