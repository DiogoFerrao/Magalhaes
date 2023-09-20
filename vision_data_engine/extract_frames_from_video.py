import os
import argparse

from datetime import datetime, timedelta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a video and name them with the time of recording"
    )
    parser.add_argument("video", help="Video to extract frames from")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--fps", type=float, default=1.0, help="fps used for extraction"
    )
    parser.add_argument("--ext", help="Extension", default="jpg")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # extract frames using ffmpeg
    cmd = f"ffmpeg -i {args.video} -qscale:v 2 -vf fps={args.fps} {args.output_dir}/frame_%06d.{args.ext}"
    print("Running:", cmd)
    os.system(cmd)
    print("Finished extracting frames.")

    # Get video time of recording
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream_tags=creation_time -of default=noprint_wrappers=1:nokey=1 {args.video}"
    print("Running:", cmd)
    time = os.popen(cmd).read().strip()
    datetime_object = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")

    # rename frames with the time of recording
    print("Renaming frames...")
    files = os.listdir(args.output_dir)
    files.sort()
    for i, f in enumerate(files):
        # only rename extracted frames
        if f.startswith("frame_"):
            frame_number = int(f.split(".")[0].split("_")[-1]) - 1
            frame_datetime = datetime_object + timedelta(
                seconds=frame_number * 1.0 / args.fps
            )
            os.rename(
                os.path.join(args.output_dir, f),
                os.path.join(
                    args.output_dir,
                    f"{frame_datetime:%Y-%m-%d__%H_%M_%S.%f+00_00}.{args.ext}",
                ),
            )
    print("Finished renaming frames.")
