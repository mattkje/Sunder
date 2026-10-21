"""Builds the Voice Changer's Seed-VC download: the parts every Seed-VC voice
shares (see models/seedvc/export.py for the model contracts).

- SeedVCWhisper.mlpackage (float16) -- Whisper-small encoder + its log-mel front end
- SeedVCVocoder.mlpackage (float32) -- BigVGAN v2 44.1 kHz

Each voice's own SeedVCDiT/SeedVCRegulator come with its .sundervoice (the
server's `sundervoice train --engine seedvc` or `sundervoice clip`). RMVPE
comes from the RVC download, which Seed-VC also needs.

Usage: PYTHONPATH=models .venv/bin/python build_coreml_seedvc.py <out_dir>
       then zip the two packages as SeedVC-v1.zip (see README).
"""

import sys

from seedvc.export import convert_shared

if __name__ == "__main__":
    convert_shared(sys.argv[1])
    print("saved", sys.argv[1])
