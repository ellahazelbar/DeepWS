import os
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import mirror_x
import moviepy.video.fx.all as vfx

def process_video(file_path, output_dir):
    basename = os.path.splitext(os.path.basename(file_path))[0]
    clip = VideoFileClip(file_path).fx(vfx.blackwhite)

    # Save the base black and white version
    bw_output = os.path.join(output_dir, f"{basename}_bw.mp4")
    clip.write_videofile(bw_output, codec="libx264", audio_codec="aac")

    # Create mirrored version
    mirrored = mirror_x(clip)
    mirrored_output = os.path.join(output_dir, f"{basename}_mirrored.mp4")
    mirrored.write_videofile(mirrored_output, codec="libx264", audio_codec="aac")

    # Create inverted version
    inverted = clip.fl_image(lambda frame: 255 - frame)
    inverted_output = os.path.join(output_dir, f"{basename}_inverted.mp4")
    inverted.write_videofile(inverted_output, codec="libx264", audio_codec="aac")

    # Create mirrored and inverted version
    mirrored_inverted = mirror_x(inverted)
    mirrored_inverted_output = os.path.join(output_dir, f"{basename}_mirrored_inverted.mp4")
    mirrored_inverted.write_videofile(mirrored_inverted_output, codec="libx264", audio_codec="aac")

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.mp4'):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing {file_path}")
            process_video(file_path, output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process MP4 files into B&W, mirrored, inverted, and both.")
    parser.add_argument("input_dir", help="Directory with input .mp4 files")
    parser.add_argument("output_dir", help="Directory to save output files")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
