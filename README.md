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
## Kmeans model 
- km_model.pt: Window size=1 sec., Hop size=0.1 sec.
- km_500_inf.pt: Window size=1 sec., Hop size=0.1 sec, Number of cluster=500

## Usage
### Extract interleaving data
```
python3 speech2unit_dir.py --audio_dir AUDIO_DIR_PATH --ext EXT --downsample 2 --lan zh --km_model KMEANS_PATH
``` 
### Extract units only
```
python3 speech2unit_dir.py --audio_dir AUDIO_DIR_PATH --ext EXT --downsample 2 --unit_only --km_model KMEANS_PATH
```
AUDIO_DIR_PATH: the directory of the audio 
KMEANS_PATH: kmeans model path

EXT: extension of the audio, ex. wav