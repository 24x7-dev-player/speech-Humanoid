from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

def create_lip_sync_images(phonemes, duration_per_phoneme=0.2):
    frames = []
    for phoneme in phonemes:
        frame = Image.new("RGB", (800, 600), color="white")
        draw = ImageDraw.Draw(frame)
        # Simple text drawing for phoneme, replace with actual mouth shapes
        draw.text((400, 300), phoneme, fill="black", anchor="mm")
        frames.extend([frame] * int(duration_per_phoneme * 24))  # 24 FPS
    return frames

def save_images_as_video(frames, output_video_file):
    clip = ImageSequenceClip([Image.fromarray(frame) for frame in frames], fps=24)
    clip.write_videofile(output_video_file, codec='libx264')

def add_audio_to_video(audio_file, video_file, output_file):
    video_clip = VideoFileClip(video_file)
    audio_clip = AudioFileClip(audio_file)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_file, codec='libx264')

# Sample usage
if __name__ == "__main__":
    text = "Hello, this is a test."
    phonemes = extract_phonemes(text)
    audio_file = "output.wav"
    
    frames = create_lip_sync_images(phonemes)
    video_file = "output_video.mp4"
    save_images_as_video(frames, video_file)
    
    final_output_file = "final_output.mp4"
    add_audio_to_video(audio_file, video_file, final_output_file)
