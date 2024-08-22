import torch
import joblib
import soundfile as sf
# from transformers import Wav2Vec2Model
from causal_hubert import DiscreteHubertEncoder, ApplyKmeans
from argparse import ArgumentParser
from faster_whisper import WhisperModel

TPS = 50

def transcribe(audio_path):
    # Read audio
    audio, sr = sf.read(audio_path)
    assert sr == 16000, "Sample rate of audio should be 16000 Hz"
    # Maybe we can use batch pipeline in faster whisper for better efficiency
    segments, info = ASR.transcribe(audio, beam_size=5, language="en", condition_on_previous_text=False, word_timestamps=True)
    return segments

def quantize(audio_path):
    feat, leng = encoder.encode(audio_path)
    ssl_units = apply_kmeans(feat)
    return [f"<|{p}|>" for p in ssl_units]
    

def combine(kms, segments):
    words = []
    for segment in segments:
        for w in segment.words:
            words.append((w.word, int(w.start * TPS)))
    for i, (w, s) in enumerate(words):
        kms.insert(i + s, ' ' + w)

    return ''.join(kms)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_audio", type=str, help="Input audio file")
    parser.add_argument("--output_path", type=str, default="tmp.txt", help="Path to save interleaving sequence")
    parser.add_argument("--device", type=str, default="cuda", help="Acceleration device")
    parser.add_argument("--km_model", type=str, default="./km_model.pt")
    parser.add_argument("--fp16", action="store_true", help="Data types for quantizing HuBERT features. Using flash_attention_2 (float16), which is faster, but sometimes results in different results")
    args = parser.parse_args()

    # Initialize Whisper model for transcribing
    ASR = WhisperModel("andybi7676/cool-whisper", device=args.device, compute_type="float16")

    # Initialize causal HuBERT and kmeans quantize module
    encoder = DiscreteHubertEncoder()
    apply_kmeans = ApplyKmeans(args.km_model, use_gpu=True)
    
    # Transcribe given audio
    segments = transcribe(args.input_audio)

    # Quantize Causal HuBERT features
    kms = quantize(args.input_audio)

    # Generate interleaving sequence
    interleave = combine(kms, segments)

    # Dump results
    with open(args.output_path, 'w') as f:
        f.write(interleave + '\n')
