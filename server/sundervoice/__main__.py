"""`python -m sundervoice serve` runs the web service; `python -m sundervoice
train ...` makes one `.sundervoice` from the command line; `python -m
sundervoice clip ...` makes a zero-shot Seed-VC voice from one reference clip."""

import argparse
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="sundervoice")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)

    train = sub.add_parser("train", help="make one .sundervoice from audio files")
    train.add_argument("files", nargs="+", type=Path)
    train.add_argument("--name", required=True)
    train.add_argument("--output", "-o", type=Path, required=True)
    train.add_argument("--engine", choices=["rvc", "seedvc"], default="rvc")
    train.add_argument("--epochs", type=int, default=None,
                       help="RVC epochs (default: suggested from the amount of audio); Seed-VC steps (default 1000)")
    train.add_argument("--isolated", action="store_true", help="files are already clean lead vocals; skip separation")
    train.add_argument("--reference", type=Path, default=None,
                       help="Seed-VC: a clean clip of the singer to use as the voice's reference (strongly recommended)")

    clip = sub.add_parser("clip", help="make a zero-shot Seed-VC voice from one clip of clean vocals")
    clip.add_argument("file", type=Path)
    clip.add_argument("--name", required=True)
    clip.add_argument("--output", "-o", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("sundervoice.web:app", host=args.host, port=args.port)
    elif args.command == "clip":
        from . import seedvc
        print(f"Saved {seedvc.from_clip(args.name, args.file.resolve(), args.output.resolve())}")
    else:
        from . import pipeline
        spec = pipeline.Spec(args.name, [f.resolve() for f in args.files], args.isolated, args.epochs, args.engine,
                             args.reference.resolve() if args.reference else None)

        def progress(stage, fraction, message):
            print(f"\r[{stage}] {fraction * 100:5.1f}%  {message}".ljust(100), end="", flush=True)

        with tempfile.TemporaryDirectory() as job_dir:
            out = pipeline.run(Path(job_dir), spec, args.output.resolve(), progress, lambda: False)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
