from argparse import ArgumentParser
from morse import MorseCodeConverter
from moviepy import ImageSequenceClip, AudioArrayClip
import numpy as np
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("-v", "--video", action="store_true", help="Generate only the video with NO audio")
    argparse.add_argument("-a", "--audio", action="store_true", help="Generate only the audio")
    argparse.add_argument("-i", "--input", required=True, help="The input file")
    argparse.add_argument("-o", "--output", required=True, help="The output file")
    argparse.add_argument("--wpm", type=int, default=20, help="Words per minute of the Morse code output")
    argparse.add_argument("--fc", "--carrier-frequency", "--audio-frequency", type=int, default=550, help="Audio frequency of the Morse code output")

    args = argparse.parse_args(args=sys.argv[1:])
    if not args.video and not args.audio:
        print("Exiting because neither audio nor video are being generated", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    if not input_path.exists():
        print("Input file does not exist, exiting", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        text = f.read()

    fs_audio = 48000 if args.video else 44100
    fs_video = 24

    converter = MorseCodeConverter(wpm=args.wpm, fs=fs_audio)
    envelope = converter.get_envelope(text)

    if args.video:
        video = MorseCodeConverter.downsample_envelope(
            envelope, fs_video, fs_audio)
        video = np.round(video * 255).astype(np.uint8)
        video = np.append(video, [0])
        video = [np.full((1,1,3), px, dtype=np.uint8) for px in video]
        video_clip = ImageSequenceClip(video, fps=fs_video).resized(new_size=(256,144))
    
    if args.audio:
        audio = converter.modulate(envelope, args.fc)
        if args.video:
            # Calculate required padding
            padding = len(video) * (fs_audio // fs_video) - len(audio)
            audio = np.pad(audio, (0, padding))
        # Array of singleton arrays, required for mono audio
        # moviepy samples at half the sample rate if the audio is mono instead
        # of stereo
        audio = np.reshape(np.repeat(audio, 2), (-1, 2))
        audio_clip = AudioArrayClip(audio, fps=fs_audio)

    if args.video:
        if args.audio:
            video_clip = video_clip.with_audio(audio_clip)
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    elif args.audio:
        audio_clip.write_audiofile(output_path, codec="aac", logger=None)
else:
    raise NotImplementedError()
