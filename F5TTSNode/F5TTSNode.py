import os
import sys
import subprocess
import re
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from TTSInference import TTSInference

class F5TTSNode:
    data_path = os.path.dirname(__file__)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        samples_path = os.path.join(cls.data_path, "samples")

        if not os.path.exists(samples_path):
            os.makedirs(samples_path)

        sample_files = [
            f for f in os.listdir(samples_path) if f.endswith(".wav")
        ]

        if not sample_files:
            sample_files = ["No .wav files found"]

        return {
            "required": {
                "sample_input": (sample_files,),
                "text": ("STRING",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    CATEGORY = "Audio Generation"

    def execute(self, sample_input, text):
        sample_path = os.path.join(self.data_path, "samples", sample_input)
        tts_inference = TTSInference(
            ref_audio=sample_path, 
            ref_text="", 
            gen_text=text,
            output_dir="/dev/shm/")
        tts_inference.infer()

        return (get_audio("/dev/shm/out.wav", start_time=0),)

NODE_CLASS_MAPPINGS = {
    "F5TTSNode": F5TTSNode
}

def get_audio(file, start_time=0, duration=0):
    ffmpeg_path = "/usr/bin/ffmpeg"
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        #TODO: scan for sample rate and maintain
        res =  subprocess.run(args + ["-f", "f32le", "-"],
                              capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(', (\\d+) Hz, (\\w+), ',res.stderr.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        raise Exception(f"VHS failed to extract audio from {file}:\n" \
                + e.stderr.decode("utf-8"))
    if match:
        ar = int(match.group(1))
        #NOTE: Just throwing an error for other channel types right now
        #Will deal with issues if they come
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1,ac)).transpose(0,1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}