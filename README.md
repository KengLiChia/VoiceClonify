# VoiceClonify

VoiceClonify is an open-source voice cloning web application built with Gradio, designed to allow users to clone their own voices.

This project is inspired by [Thorsten Voice](https://www.youtube.com/watch?v=bJjzSo_fOS8&ab_channel=Thorsten-Voice).

## Prerequisites

1. 🐸 [TTS](https://github.com/coqui-ai/TTS) supports [Python >=3.7 <3.11.0](https://www.python.org/downloads/) and tested on Ubuntu 18.10, 19.10, 20.10, Windows 10, 11.
2. [eSpeak-ng](https://github.com/espeak-ng/espeak-ng/releases/tag/1.51)
3. [Microsoft Build Tools for C++](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Activate desktop development with C++)
4. [Audacity](https://www.audacityteam.org/download/)
5. [FFmpeg](https://www.ffmpeg.org/download.html)

## Installation

1. Create a virtual environment (optional but recommended)

   ```bash
   python -m venv .
   ```

2. Activate the virtual environment (on Windows)

   ```bash
   .\Scripts\activate
   ```

3. Upgrade setuptools and wheel

   ```bash
   pip install setuptools wheel -U
   ```

4. Install dependency [PyTorch](https://pytorch.org/get-started/locally/) (for CUDA support)

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. Under the folder TTS-0.20.3,

   ```bash
   cd TTS-0.20.3
   pip install -e .
   ```

6. Build and run for creating datasets

   ```bash
   cd mimic-recording-studio-master
   docker-compose up
   ```

7. Run `MRS2LJSpeech.py` file to generate dataset created from previous step.

   ```bash
   python ./MRSLJSpeech.py
   ```

8. Ensure datasets wavs are converted from WAVs to Mono, 22050 using Audacity (Manage Macros)
   - 01 Stereo To Mono
   - 02 Set Project Rate= "22050"
   - 03 Export as WAV
   - 04 -END-

## Tensorboard

Install TensorBoard (optional for training)

```bash
pip install tensorboard
```

Under the training folder,

```bash
tensorboard --logdir=.
```

## Training

```bash
python ..\train_vits_win_espeak.py
```

## Finetuning

```bash
python .\train_fast_pitch.py --restore_path "C:\Users\krist\AppData\Local\tts\tts_models--en--ljspeech--fast_pitch\model_file.pth" --coqpit.run_name "fast_pitch_finetuning"
```

## Resume training

```bash
python ..\train_fast_pitch.py --continue_path .
```

## Test trained voice

Below is an example to test your trained voice model.

### Fast Pitch

```bash
python C:\Users\krist\Documents\GitHub\VoiceClonify\TTS-0.20.3\TTS\bin\synthesize.py --config_path C:\Users\krist\Documents\GitHub\VoiceClonify\kengli_training_space\fast_pitch_finetuning-June-02-2024_01+35AM-0ae693a\config.json --model_path C:\Users\krist\Documents\GitHub\VoiceClonify\kengli_training_space\fast_pitch_finetuning-June-02-2024_01+35AM-0ae693a\checkpoint_290000.pth --text "The price is fair for a good antique clock." --out_path C:\Users\krist\Documents\GitHub\VoiceClonify\audio_files\fastpitch.wav
```

### Fast Speech

```bash
python C:\Users\krist\Documents\GitHub\VoiceClonify\TTS-0.20.3\TTS\bin\synthesize.py --config_path C:\Users\krist\Documents\GitHub\VoiceClonify\kengli_training_space\fast_speech_finetuning-May-08-2024_11+26PM-310580b\config.json --model_path C:\Users\krist\Documents\GitHub\VoiceClonify\kengli_training_space\fast_speech_finetuning-May-08-2024_11+26PM-310580b\checkpoint_300000.pth --text "The price is fair for a good antique clock." --out_path C:\Users\krist\Documents\GitHub\VoiceClonify\audio_files\fastspeech.wav
```

### GlowTTS

```bash
python C:\Users\krist\Documents\GitHub\VoiceClonify\TTS-0.20.3\TTS\bin\synthesize.py --config_path C:\Users\krist\Documents\GitHub\VoiceClonify\kengli_training_space\my_glow_tts_finetuning-April-07-2024_02+58AM-310580b\config.json --model_path C:\Users\krist\Documents\GitHub\VoiceClonify\kengli_training_space\my_glow_tts_finetuning-April-07-2024_02+58AM-310580b\checkpoint_880000.pth --text "The price is fair for a good antique clock." --out_path C:\Users\krist\Documents\GitHub\VoiceClonify\audio_files\glowtts.wav
```

### VITS

```bash
python C:\Users\krist\Documents\GitHub\VoiceClonify\TTS-0.20.3\TTS\bin\synthesize.py --config_path C:\Users\krist\Documents\GitHub\VoiceClonify\kengli_training_space\vits-espeak-phonemizer\config.json --model_path C:\Users\krist\Documents\GitHub\VoiceClonify\kengli_training_space\vits-espeak-phonemizer\checkpoint_60000.pth --text "The price is fair for a good antique clock." --out_path C:\Users\krist\Documents\GitHub\VoiceClonify\audio_files\vits.wav
```

### TTS

Run in `kengli_training_space` folder.

## Install Gradio

```bash
pip install gradio
```

## Setup Voice Clonify

1. Install the dependencies

```bash
  pip install -r requirements.txt
```

2. Run the backend server

```bash
  python backend/app.py
```

3. Run the Gradio frontend

```bash
   python frontend/gradio_frontend.py
```

4. Open the Gradio link in your browser to interact with the applicaton.

## License

[MIT](https://choosealicense.com/licenses/mit/)
