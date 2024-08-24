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


def transcribe(audio_path):
    # Read audio
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    segments, info = ASR.transcribe(audio, beam_size=5, language=args.lan, condition_on_previous_text=False, word_timestamps=True)
    return segments

def quantize(audio_path, downsample):
    feat = encoder.encode(audio_path)
    ssl_units = apply_kmeans(feat)
    return [f"<|{p}|>" for p in ssl_units][::downsample]

def combine(kms, segments):
    words = []
    for segment in segments:
        for w in segment.words:
            words.append((w.word, int(w.start * TPS)))
    for i, (w, s) in enumerate(words):
        # print(w, s)
        kms.insert(i + s, ' ' + w)

    return ''.join(kms)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--audio_dir", type=str, help="Audio dir")
    parser.add_argument("--ext", type=str, help="Wave format", default='wav')
    parser.add_argument("--downsample", type=int, help="Downsample ratio", default=2)
    parser.add_argument("--lan", type=str, help="Language code", default='zh')
    parser.add_argument("--device", type=str, default="cuda", help="Acceleration device")
    parser.add_argument("--km_model", type=str, default="./km_model.pt")
    parser.add_argument("--fp16", action="store_true", help="Data types for quantizing HuBERT features. Using flash_attention_2 (float16), which is faster, but sometimes results in different results")
    args = parser.parse_args()

    TPS = 50/args.downsample
    # Initialize Whisper model for transcribing
    ASR = WhisperModel("andybi7676/cool-whisper", device=args.device, compute_type="float16")

    # Initialize causal HuBERT and kmeans quantize module
    encoder = StreamingHubertEncoder()
    apply_kmeans = ApplyKmeans(args.km_model, use_gpu=True)
    
    file_lists = list(glob.glob(os.path.join(args.audio_dir, f'**/*.{args.ext}'), recursive=True))
    print(f"Generate Interleaving Data for {len(file_lists)} files.")

    for audio_path in tqdm(file_lists):
        # Transcribe given audio
        segments = transcribe(audio_path)
        # Quantize Causal HuBERT features
        kms = quantize(audio_path, args.downsample)
        # Generate interleaving sequence
        interleave = combine(kms, segments)
        # Output path
        output_path = os.path.splitext(audio_path)[0] + ".txt"
        # Dump results
        with open(output_path, 'w') as f:
            f.write(interleave + '\n')