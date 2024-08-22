# Speech to Interleaving Unit

## Install
python >= 3.8
```
pip install -r requirements
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