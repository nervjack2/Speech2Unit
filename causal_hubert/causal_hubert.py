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


class DiscreteHubertEncoder():
    def __init__(self, batch_size=16, device="cuda"):
        model_path = "TencentGameMate/chinese-hubert-base"

        self.batch_size = batch_size
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = HubertModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()


    def batch_encode(self, file_list):
        feats, lens = [], []
        for i in tqdm(range(0, len(file_list), self.batch_size)):
            start_idx = i
            end_idx = min(i+self.batch_size, len(file_list))
            wavs = []
            for j in range(start_idx, end_idx):
                wav, sr = sf.read(file_list[j])
                wavs.append(torch.from_numpy(wav))

            batch_feats, batch_lens = self._encode(wavs)

            btz = batch_feats.shape[0]
            feats.extend([batch_feats[j, :batch_lens[j], :].numpy() for j in range(btz)])
            lens.extend(batch_lens)

            torch.cuda.empty_cache()

            # if len(feats) >= SHARD_SIZE:
            #     torch.save({
            #         "feats": feats[:SHARD_SIZE], "lens": lens[:SHARD_SIZE]
            #     }, f"km_data_new/yt-data-{shard_id}.pt")
            #     shard_id += 1
            #     feats = feats[SHARD_SIZE:]
            #     lens = lens[SHARD_SIZE:]
            #
        # torch.save({
        #     "feats": feats, "lens": lens
        # }, f"km_data_new/yt-data-{shard_id}.pt")

        return feats, lens


    def encode(self, audio_input, sr=16000):
        """
        Encode one audio into discrete Hubert units.

        Parameters:
            audio_input(str, np.ndarray): can be path string or numpy array
            sr(int): sampling rate for audio_input
        """

        if type(audio_input) == str:
            wav, sr = sf.read(audio_input)
        elif type(audio_input) == np.ndarray:
            if sr != 16000:
                audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)
            wav = audio_input
        else:
            raise NotImplementedError

        feats, lens = self._encode([wav])
        return feats[0], lens[0]


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

