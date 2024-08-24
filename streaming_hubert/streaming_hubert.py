import torch.nn.functional as F
import soundfile as sf
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from tqdm import tqdm
import torch
import numpy as np
import librosa

SHARD_SIZE = 100
HOP_LENGTH = 1600  # 100ms
WIN_LENGTH = 16000  # 1s

class StreamingHubertEncoder():
    def __init__(self, batch_size=16, device="cuda"):
        model_path = "TencentGameMate/chinese-hubert-base"
        self.batch_size = batch_size
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = HubertModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()


    def batch_encode(self, audio_list):
        """
        Encode a list of audio

        Parameters:
            audio_list(list(str) or list(tensor)): list of audio in path or tensor format

        Returns:
            feat: a list of Hubert representations for each file
        """
        feats = []
        # shard_id = 0
        for i in range(len(audio_list)):
            if type(audio_list[i]) == str:
                wav, sr = sf.read(audio_list[i])
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                wav = torch.from_numpy(wav)
            elif type(audio_list[i]) == torch.tensor:
                wav = audio_list[i].squeeze()
                assert len(audio_list[i].shape) == 1, "You should use single channel audio"
            else:
                raise NotImplementedError
            wav_slices = []
            wav_feat = []
            for i in range(HOP_LENGTH, wav.shape[0], HOP_LENGTH):
                start_pos = max(i-WIN_LENGTH, 0)
                wav_slices.append(wav[start_pos:i])

                if len(wav_slices) >= self.batch_size or i+HOP_LENGTH >= wav.shape[0]:
                    batch_feats, batch_lens = self._encode(wav_slices)
                    for bi in range(len(batch_feats)):
                        wav_feat.extend(batch_feats[bi][:batch_lens[bi]][-5:])
                    wav_slices = []
                    torch.cuda.empty_cache()

            wav_feat = torch.vstack(wav_feat)
            feats.append(wav_feat)

            # torch.save({
            #     "feats": torch.vstack(wav_feat), "lens": []
            # }, f"km_data_new/soundon-data-{shard_id}.pt")
            # shard_id += 1

        return feats


    def encode(self, audio_input):
        feats = self.batch_encode([audio_input])
        return feats[0]


    def _encode(self, wavs):
        """
        Encode list of audio into Hubert features

        Parameters:
            wavs: list of np.ndarray

        Returns:
            feats: list of torch.tensor, representing the L6 Hubert features
            lens: list of integer, representing the lengths of the features
        """
        is_batch = (len(wavs) > 1)
        wavs = [torch.tensor(wav, dtype=torch.float32) for wav in wavs]
        max_len = max(wav.shape[0] for wav in wavs)
        wavs_padded = [F.pad(wav, (0, max_len - wav.shape[0])) for wav in wavs]
        wavs_padded = torch.vstack(wavs_padded).squeeze()

        input_values = self.feature_extractor(wavs_padded, return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(self.device)
        if is_batch:
            input_values = input_values.squeeze()
        outputs = self.model(input_values, attention_mask=torch.ones(input_values.shape[0]).to(self.device), output_hidden_states=True)
        feats = outputs.hidden_states[6].detach().cpu()
        lens = [(l.shape[0]-80)//320 for l in wavs]

        return feats, lens

