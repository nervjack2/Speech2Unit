# Speech to Interleaving Unit
Transform audio into interleaving sequnce. 
We use COOLWhisper for ASR transcription, CausalHuBERT+Kmeans for speech units. 

See audio_dir/ for examples of interleaving results.
## Install
python == 3.8
```
git clone https://github.com/nervjack2/Speech2Unit.git
cd Speech2Unit
pip install -r requirements.txt
git clone https://github.com/huggingface/transformers.git
cp modeling_hubert.py transformers/src/transformers/models/hubert/modeling_hubert.py 
cd transformers
pip install .
pip install faster-whisper
```

## Usage
```
python3 speech2unit_dir.py --audio_dir AUDIO_DIR_PATH --ext EXT
``` 
AUDIO_DIR_PATH: the directory of the audio 

EXT: extension of the audio, ex. wav
