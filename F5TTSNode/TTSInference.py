import codecs
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
import torch
import torchaudio
import tqdm
from cached_path import cached_path
from einops import rearrange
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from model import CFM, DiT, MMDiT, UNetT
from model.utils import (convert_char_to_pinyin, get_tokenizer,
                         load_checkpoint, save_spectrogram)

class TTSInference:
    """
    A class for performing text-to-speech (TTS) inference using advanced batch processing.

    Attributes:
    ----------
    config : dict
        Configuration parameters loaded from a TOML file.
    ref_audio : str
        Path to the reference audio file.
    ref_text : str
        Reference text associated with the reference audio.
    gen_text : str
        Text to generate speech for.
    gen_file : str
        Path to a file containing text to generate speech for.
    output_dir : str
        Directory where output files will be saved.
    model_name : str
        Name of the TTS model to use ("F5-TTS" or "E2-TTS").
    remove_silence : bool
        Flag indicating whether to remove silence from generated audio.
    wave_path : Path
        Path to save the generated waveform.
    spectrogram_path : Path
        Path to save the generated spectrogram.
    vocos_local_path : str
        Path to the local vocoder checkpoint.
    device : str
        Device to use for computation ("cuda", "mps", or "cpu").
    target_sample_rate : int
        Sample rate for the generated audio.
    n_mel_channels : int
        Number of mel frequency channels.
    hop_length : int
        Hop length for mel spectrogram calculation.
    target_rms : float
        Target root mean square (RMS) value for audio normalization.
    nfe_step : int
        Number of steps for neural function estimation.
    cfg_strength : float
        Configuration strength for inference.
    ode_method : str
        Method to use for ODE (ordinary differential equation) solving.
    sway_sampling_coef : float
        Coefficient for sway sampling.
    speed : float
        Speed factor for audio generation.
    fix_duration : float or None
        Fixed duration for generated audio segments.
    F5TTS_model_cfg : dict
        Configuration parameters for the F5-TTS model.
    E2TTS_model_cfg : dict
        Configuration parameters for the E2-TTS model.
    """
    
    def __init__(self, config_path="inference-cli.toml", model_name=None, ref_audio=None, ref_text="666", gen_text=None,
                 gen_file=None, output_dir=None, remove_silence=False, load_vocoder_from_local=False):
        self.config = tomli.load(open(config_path, "rb"))

        self.ref_audio = ref_audio if ref_audio else self.config["ref_audio"]
        self.ref_text = ref_text if ref_text != "666" else self.config["ref_text"]
        self.gen_text = gen_text if gen_text else self.config["gen_text"]
        self.gen_file = gen_file if gen_file else self.config["gen_file"]
        if self.gen_file:
            self.gen_text = codecs.open(self.gen_file, "r", "utf-8").read()
        self.output_dir = output_dir if output_dir else self.config["output_dir"]
        self.model_name = model_name if model_name else self.config["model"]
        self.remove_silence = remove_silence if remove_silence else self.config["remove_silence"]
        self.wave_path = Path(self.output_dir) / "out.wav"
        self.spectrogram_path = Path(self.output_dir) / "out.png"
        self.vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        if load_vocoder_from_local:
            print(f"Load vocos from local path {self.vocos_local_path}")
            self.vocos = Vocos.from_hparams(f"{self.vocos_local_path}/config.yaml")
            state_dict = torch.load(f"{self.vocos_local_path}/pytorch_model.bin", map_location=self.device)
            self.vocos.load_state_dict(state_dict)
            self.vocos.eval()
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

        print(f"Using {self.device} device")

        # Settings
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1
        self.nfe_step = 32
        self.cfg_strength = 2.0
        self.ode_method = "euler"
        self.sway_sampling_coef = -1.0
        self.speed = 1.0
        self.fix_duration = None

        # Load model configurations
        self.F5TTS_model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        self.E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

    def load_model(self, repo_name, exp_name, model_cls, model_cfg, ckpt_step):
        """
        Load the TTS model based on the provided parameters.

        Parameters:
        -----------
        repo_name : str
            Repository name containing the model.
        exp_name : str
            Experiment name associated with the model.
        model_cls : class
            The class representing the model to be loaded.
        model_cfg : dict
            Configuration parameters for the model.
        ckpt_step : int
            The checkpoint step to load the model from.

        Returns:
        --------
        model : torch.nn.Module
            The loaded TTS model.
        """
        ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"
        if not Path(ckpt_path).exists():
            ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
        model = CFM(
            transformer=model_cls(
                **model_cfg, text_num_embeds=vocab_size, mel_dim=self.n_mel_channels
            ),
            mel_spec_kwargs=dict(
                target_sample_rate=self.target_sample_rate,
                n_mel_channels=self.n_mel_channels,
                hop_length=self.hop_length,
            ),
            odeint_kwargs=dict(
                method=self.ode_method,
            ),
            vocab_char_map=vocab_char_map,
        ).to(self.device)

        model = load_checkpoint(model, ckpt_path, self.device, use_ema=True)
        return model

    def chunk_text(self, text, max_chars=135):
        """
        Split input text into chunks of a specified maximum length.

        Parameters:
        -----------
        text : str
            The input text to be split.
        max_chars : int
            The maximum number of characters allowed per chunk.

        Returns:
        --------
        chunks : list of str
            List of text chunks.
        """
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[;:,.!?])\s+|(?<=[\uff1b\uff1a\uff0c\u3002\uff01\uff1f])', text)

        for sentence in sentences:
            if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
                current_chunk += sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def infer_batch(self, ref_audio, ref_text, gen_text_batches, model_name, remove_silence, cross_fade_duration=0.15):
        """
        Perform inference in batches to generate speech.

        Parameters:
        -----------
        ref_audio : tuple
            Tuple containing the reference audio tensor and its sample rate.
        ref_text : str
            Reference text for the audio.
        gen_text_batches : list of str
            List of text batches to generate speech for.
        model_name : str
            Name of the model to use for inference ("F5-TTS" or "E2-TTS").
        remove_silence : bool
            Flag indicating whether to remove silence from the generated audio.
        cross_fade_duration : float
            Duration for cross-fading between generated segments.

        Returns:
        --------
        None
        """
        if model_name == "F5-TTS":
            ema_model = self.load_model(model_name, "F5TTS_Base", DiT, self.F5TTS_model_cfg, 1200000)
        elif model_name == "E2-TTS":
            ema_model = self.load_model(model_name, "E2TTS_Base", UNetT, self.E2TTS_model_cfg, 1200000)

        audio, sr = ref_audio
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < self.target_rms:
            audio = audio * self.target_rms / rms
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(self.device)

        generated_waves = []
        spectrograms = []

        for i, gen_text in enumerate(tqdm.tqdm(gen_text_batches)):
            if len(ref_text[-1].encode('utf-8')) == 1:
                ref_text = ref_text + " "
            text_list = [ref_text + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)

            ref_audio_len = audio.shape[-1] // self.hop_length
            zh_pause_punc = r"\uff0c\u3002\uff1b\uff1a\uff01\uff1f"
            ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
            gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / self.speed)

            with torch.inference_mode():
                generated, _ = ema_model.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=self.nfe_step,
                    cfg_strength=self.cfg_strength,
                    sway_sampling_coef=self.sway_sampling_coef,
                )

            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
            generated_wave = self.vocos.decode(generated_mel_spec.cpu())
            if rms < self.target_rms:
                generated_wave = generated_wave * rms / self.target_rms

            generated_wave = generated_wave.squeeze().cpu().numpy()
            
            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

        if cross_fade_duration <= 0:
            final_wave = np.concatenate(generated_waves)
        else:
            final_wave = generated_waves[0]
            for i in range(1, len(generated_waves)):
                prev_wave = final_wave
                next_wave = generated_waves[i]

                cross_fade_samples = int(cross_fade_duration * self.target_sample_rate)
                cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                if cross_fade_samples <= 0:
                    final_wave = np.concatenate([prev_wave, next_wave])
                    continue

                prev_overlap = prev_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]

                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)

                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                new_wave = np.concatenate([
                    prev_wave[:-cross_fade_samples],
                    cross_faded_overlap,
                    next_wave[cross_fade_samples:]
                ])

                final_wave = new_wave

        with open(self.wave_path, "wb") as f:
            sf.write(f.name, final_wave, self.target_sample_rate)
            if remove_silence:
                aseg = AudioSegment.from_file(f.name)
                non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    non_silent_wave += non_silent_seg
                aseg = non_silent_wave
                aseg.export(f.name, format="wav")
            print(f.name)

        combined_spectrogram = np.concatenate(spectrograms, axis=1)
        save_spectrogram(combined_spectrogram, self.spectrogram_path)
        print(self.spectrogram_path)

    def infer(self, cross_fade_duration=0.15):
        """
        Perform inference to generate speech from text.

        Parameters:
        -----------
        cross_fade_duration : float
            Duration for cross-fading between generated segments.

        Returns:
        --------
        None
        """
        print(self.gen_text)
        print("Converting audio...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            aseg = AudioSegment.from_file(self.ref_audio)

            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave

            audio_duration = len(aseg)
            if audio_duration > 15000:
                print("Audio is over 15s, clipping to only first 15s.")
                aseg = aseg[:15000]
            aseg.export(f.name, format="wav")
            ref_audio = f.name

        if not self.ref_text.strip():
            print("No reference text provided, transcribing reference audio...")
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16,
                device=self.device,
            )
            self.ref_text = pipe(
                ref_audio,
                chunk_length_s=30,
                batch_size=128,
                generate_kwargs={"task": "transcribe"},
                return_timestamps=False,
            )["text"].strip()
            print("Finished transcription")
        else:
            print("Using custom reference text...")

        if not self.ref_text.endswith(". ") and not self.ref_text.endswith("。"):
            if self.ref_text.endswith("."):
                self.ref_text += " "
            else:
                self.ref_text += ". "

        audio, sr = torchaudio.load(ref_audio)
        max_chars = int(len(self.ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
        gen_text_batches = self.chunk_text(self.gen_text, max_chars=max_chars)
        print('ref_text', self.ref_text)
        for i, gen_text in enumerate(gen_text_batches):
            print(f'gen_text {i}', gen_text)
        
        print(f"Generating audio using {self.model_name} in {len(gen_text_batches)} batches, loading models...")
        return self.infer_batch((audio, sr), self.ref_text, gen_text_batches, self.model_name, self.remove_silence, cross_fade_duration)


# Example usage
if __name__ == "__main__":
    tts_inference = TTSInference(
        ref_audio="samples/RodSerling.wav", 
        ref_text="", 
        gen_text="Imagine a world where you can do anything you want. A world where you can be anyone you want to be. A world where you can go anywhere you want to go. A world where you can see anything you want to see. A world where you can feel anything you want to feel. A world where you can experience anything you want to experience. A world where you can create anything you want to create. A world where you can dream anything you want to dream. A world where you can live any life you want to live. A world where you can be free. A world where you can be happy. A world where you can be yourself. A world without limits.",
        output_dir="/dev/shm/")
    tts_inference.infer()
