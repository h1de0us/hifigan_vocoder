import logging
from typing import List
import torch

from src.utils.preprocessing import MelSpectrogram, MelSpectrogramConfig

logger = logging.getLogger(__name__)


def pad_sequence(items: List[torch.Tensor], pad_value: float = 0.0):
    '''
    Pad sequence to max_len by using torch.rnn.pad_sequence
    '''
    padded_items = torch.nn.utils.rnn.pad_sequence(
        items, batch_first=True, padding_value=pad_value
    )
    return padded_items



def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # print(dataset_items[0]["audio"].shape) # (1, T)

    audios = torch.as_tensor(pad_sequence([item["audio"].squeeze(0) for item in dataset_items]))

    # print(audios.shape) # (batch_size, max_seq_len)
    melspec = MelSpectrogram(MelSpectrogramConfig())
    spectrograms = melspec(audios)
    # print('spectrograms.shape', spectrograms.shape)  # (batch_size, n_mels, max_seq_len // hop_length)

    return {
        "audio": audios.unsqueeze(1), # return squeezed dimension, (batch_size, n_channels, time)
        # "audio_path": [item["audio_path"] for item in dataset_items],
        "duration" : torch.as_tensor([item["duration"] for item in dataset_items]),
        "spectrogram": spectrograms, 
    }
    