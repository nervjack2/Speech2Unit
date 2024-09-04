import torch
import joblib
import glob
import librosa
import os 
import soundfile as sf
from streaming_hubert import StreamingHubertEncoder, ApplyKmeans
from argparse import ArgumentParser
from faster_whisper import WhisperModel
from tqdm import tqdm 

def quantize_batch(batch_audio_path, downsample):
    feats = encoder.batch_encode(batch_audio_path)
    ssl_units = [apply_kmeans(feat) for feat in feats]
    return [[f"<|{p}|>" for p in seq][::downsample] for seq in ssl_units]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--audio_dir", type=str, help="Audio dir")
    parser.add_argument("--ext", type=str, help="Wave format", default='wav')
    parser.add_argument("--downsample", type=int, help="Downsample ratio", default=2)
    parser.add_argument("--device", type=str, default="cuda", help="Acceleration device")
    parser.add_argument("--km_model", type=str, default="./km_500_inf.pt")
    parser.add_argument("--bsz", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    TPS = 50/args.downsample

    # Initialize causal HuBERT and kmeans quantize module
    encoder = StreamingHubertEncoder()
    apply_kmeans = ApplyKmeans(args.km_model, use_gpu=True)
    
    file_lists = list(glob.glob(os.path.join(args.audio_dir, f'**/*.{args.ext}'), recursive=True))
    print(f"Generate Interleaving Data for {len(file_lists)} files.")
    file_lists = [file_lists[i:i + args.bsz] for i in range(0, len(file_lists), args.bsz)]

    for batch_audio_path in tqdm(file_lists):
        # Quantize Causal HuBERT features
        kms = quantize_batch(batch_audio_path, args.downsample)
        for idx, seq in enumerate(kms):
            # Output path
            output_path = os.path.splitext(batch_audio_path[idx])[0] + ".txt"
            # Dump results
            with open(output_path, 'w') as f:
                f.write(" ".join(seq) + '\n')