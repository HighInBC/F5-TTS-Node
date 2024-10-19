from TTSInference import TTSInference
import os

class FSTTSNode:
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

        with open("/dev/shm/out.wav", "rb") as f:
            audio_data = f.read()

        return audio_data

NODE_CLASS_MAPPINGS = {
    "FSTTSNode": FSTTSNode
}
