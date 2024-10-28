import torch
from torch.utils.data import Dataset

class PhonemeAudioDataset(Dataset):
    def __init__(self, phoneme_data, audio_data):
        self.phoneme_data = phoneme_data
        self.audio_data = audio_data

    def __len__(self):
        return len(self.phoneme_data)

    def __getitem__(self, idx):
        phonemes = self.phoneme_data[idx]
        audio = self.audio_data[idx]
        return phonemes, audio
