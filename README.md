# VoiceClonify

VoiceClonify is an open-source voice cloning application designed to allow users to clone their own voices or mimic any desired voice for text-to-speech generation.

This project is inspired by [Thorsten Voice](https://www.youtube.com/watch?v=bJjzSo_fOS8&ab_channel=Thorsten-Voice).

## Features

### 1. Voice Cloning

- **User-friendly Interface:** Navigate through the application with an intuitive design that simplifies the voice cloning process.

### 2. Real-time Visualization

- **Training Visualization:** Visualize the process of model training your voice in real-time.

### 3. Voice Comparison

- **Similarity Assessment:** Utilize algorithms to determine similarities and differences between the original and cloned voices.

### 4. Text-to-Speech Generation

- **Versatile Text Input:** Input any text and listen to the synthesized speech using the cloned voice.

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
    pip3 install torch torchvision torchaudio
    OR
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

5. Under the folder TTS-0.20.3,
    ```bash
    cd TTS-0.20.3
    pip install -e .
    ```

6. Build and run for creating datasets
    ```bash
    cd mimic-recording-studio
    docker-compose up
    ```

7. To convert WAVs to Mono, 22050 using Audacity (Manage Macros)
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
## License
[MIT](https://choosealicense.com/licenses/mit/)