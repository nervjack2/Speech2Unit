# Speech to Interleaving Unit
Transform audio into interleaving sequnce. 
We use COOLWhisper for ASR transcription, StreamingHuBERT for speech units. 

See audio_dir/ for examples of interleaving results.
## Install (Important)
python == 3.8
```
git clone https://github.com/anthony-wss/streaming-hubert-encoder
cd streaming-hubert-encoder
pip install -r requirements
pip install .
pip install transformers
```

## Usage
```
python3 speech2unit_dir.py --audio_dir AUDIO_DIR_PATH --ext EXT --downsample 2
``` 
AUDIO_DIR_PATH: the directory of the audio 

EXT: extension of the audio, ex. wav
