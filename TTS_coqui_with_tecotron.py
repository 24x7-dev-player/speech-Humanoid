# !pip install TTS


from TTS.api import TTS

def text_to_speech(text, output_audio_file):
    # Initialize TTS with the desired model
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
    # Convert text to speech
    tts.tts_to_file(text=text, file_path=output_audio_file)
    print(f"Audio saved to {output_audio_file}")

# Sample usage
if __name__ == "__main__":
    text = "Hello, this is a test."
    audio_file = "output.wav"
    text_to_speech(text, audio_file)
