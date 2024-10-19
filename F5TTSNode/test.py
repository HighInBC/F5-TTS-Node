from TTSInference import TTSInference

tts_inference = TTSInference(
    ref_audio="samples/RodSerling.wav", 
    ref_text="", 
    gen_text="Imagine a world where you can do anything you want. A world where you can be anyone you want to be. A world where you can go anywhere you want to go. A world where you can see anything you want to see. A world where you can feel anything you want to feel. A world where you can experience anything you want to experience. A world where you can create anything you want to create. A world where you can dream anything you want to dream. A world where you can live any life you want to live. A world where you can be free. A world where you can be happy. A world where you can be yourself. A world without limits.",
    output_dir="/dev/shm/")
tts_inference.infer()

# load the file /dev/shm/out.wav into audio_data
with open("/dev/shm/out.wav", "rb") as f:
    audio_data = f.read()
