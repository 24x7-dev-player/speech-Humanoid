import bpy

def animate_lip_sync(audio_file, phonemes_file):
    # Load your 3D model
    bpy.ops.import_scene.obj(filepath="/path/to/your/model.obj")
    model = bpy.context.selected_objects[0]
    
    # Assume you have shape keys named after phonemes
    shape_keys = model.data.shape_keys.key_blocks

    # Example of setting a shape key value
    def set_shape_key_value(key_name, value):
        if key_name in shape_keys:
            shape_keys[key_name].value = value

    # Read phonemes from file
    with open(phonemes_file, 'r') as f:
        phonemes = f.read().splitlines()

    # Create an animation for the phonemes
    frame_number = 1
    for phoneme in phonemes:
        bpy.context.scene.frame_set(frame_number)
        set_shape_key_value(phoneme, 1.0)  # Activate the phoneme shape key
        model.keyframe_insert(data_path='["%s"]' % phoneme, frame=frame_number)
        frame_number += 5  # Move to the next frame
        set_shape_key_value(phoneme, 0.0)  # Deactivate the phoneme shape key
        model.keyframe_insert(data_path='["%s"]' % phoneme, frame=frame_number)
        frame_number += 5

    # Add the audio to the scene
    bpy.ops.sequencer.sound_strip_add(filepath=audio_file, frame_start=1)
    
    # Render the animation
    bpy.context.scene.render.filepath = '/path/to/output/animation.mp4'
    bpy.ops.render.render(animation=True)

# Example usage (inside Blender's Python environment)
audio_file = "/path/to/output.wav"
phonemes_file = "/path/to/phonemes.txt"
animate_lip_sync(audio_file, phonemes_file)
