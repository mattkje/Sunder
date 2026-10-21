# Sunder Voice Server

A standalone Linux service that trains voice models for Sunder's Voice Changer.
Upload songs (or vocals you've already isolated), and it produces a
`.sundervoice` file you import in Sunder → Voice Changer → **Import…**.
It's completely separate from the app: nothing here talks to a Mac.

## What a job does

1. **Separation** (skipped if you tick "already isolated lead vocals"). Each song
   goes through three Roformer models via
   [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator):
   - vocals/instrumental split (`mel_band_roformer_vocals_becruily`)
   - lead vs backing vocals (`mel_band_roformer_karaoke_becruily`), keeping the lead only
   - de-reverb (`dereverb_mel_band_roformer_anvuew`), since RVC trains best on dry vocals
   Then the job trains one of two models, picked when you create it:
   **Seed-VC** (natural, close to the real singer; steps 2-4 below differ, see
   "Seed-VC" further down) or **RVC**.
2. **Preparation.** This is RVC's own preprocessing: silence slicing, 3.7 s segments,
   ContentVec features and RMVPE pitch.
3. **Training.** RVC v2 40 kHz is fine-tuned from the pretrained base with the same
   recipe as the app. Checkpoints are saved every 5 epochs. The number of epochs
   defaults to about 8k optimizer steps (50–300 epochs, depending on how much audio you give it).
4. **Export.** It writes `G.safetensors` (in the MLX layout, float16), the retrieval index,
   and `profile.json`, zipped into the format Sunder's importer expects.

A stopped job, or one that was interrupted by a restart, resumes from its last
checkpoint. Jobs run one at a time.

## Seed-VC

Seed-VC jobs cut the lead vocals into 2-25 s clips and fine-tune Seed-VC's
singing model on them (1000 steps by default; the "Steps" field overrides it),
checkpointing every 100 steps. The export converts the voice's model to CoreML
and picks 25 s of the vocals as its reference. Sunder needs its Seed-VC
models (Settings → Models) to use these voices. On the CPU, Seed-VC training is
very slow; use a GPU or run it on a Mac (see "Without Docker").

A voice can also be made without training, from one clean clip of the singer
(about 25 s is used):

```sh
docker compose run --rm sundervoice python -m sundervoice clip /data/in/clip.flac \
  --name "My Voice" -o /data/my-voice.sundervoice
```

## Run it (Docker)

You need Docker with Compose v2.

```sh
cd server
docker compose up -d --build
```

Then open `http://<server>:8765`. Everything is stored in `server/data/`:
- `models/`: the weights, about 2 GB, downloaded on the first job
- `jobs/<id>/`: each job's lead vocals and the finished `voice.sundervoice`

The image builds for the CPU by default. To use an NVIDIA GPU, uncomment the build
`args` and the `deploy:` block in `docker-compose.yml`. That also needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. A GPU makes training hours instead of a day or more.

There's no login, so keep it on your LAN and don't expose it to the internet.

## Instrumentals

The **Instrumentals** tab browses a Jellyfin music library. Tick whole albums or single songs and press
**Generate Instrumental**: each song goes through the same vocals/instrumental Roformer as the voice pipeline,
and two files are written next to the original:

- `<original name> [Instrumental].flac`
- `<original name> [Vocals].flac`

Both keep the original's tags and cover art, with ` [Instrumental]` / ` [Vocals]` added to the title, and
Jellyfin is asked to rescan the album. Coral hides these files and uses them for its vocal slider instead;
other Jellyfin apps show them as extra tracks. Songs that already have both files are skipped.

Setup (see the commented lines in `docker-compose.yml`):
- `JELLYFIN_URL`: the Jellyfin server, reachable from this container.
- `JELLYFIN_API_KEY`: from Jellyfin's Dashboard → API Keys. It's used to read the library and each track's file path.
- Mount the music library **read-write**. If Jellyfin sees it at a different path than this container,
  set `JELLYFIN_PATH_MAP` to `<Jellyfin path>=<container path>` (comma-separate several).

Instrumental jobs share the queue with voice training, so they wait for a running training job.

## Command line

```sh
docker compose run --rm sundervoice python -m sundervoice train \
  --name "My Voice" -o /data/my-voice.sundervoice /data/in/*.flac
# add --isolated for vocals that are already clean; --engine seedvc for Seed-VC;
# --epochs N to override (Seed-VC: steps)
```

## Without Docker

```sh
python3 -m venv .venv && .venv/bin/pip install torch "audio-separator[cpu]" -r requirements.txt
PYTHONPATH=../tools/convert/models SUNDERVOICE_DATA=./data SUNDERVOICE_MODELS=./data/models \
  .venv/bin/python -m sundervoice serve
```

(`ffmpeg` must be installed.)

## Licenses

- The RVC code and weights are MIT.
- ContentVec is MIT.
- The becruily and anvuew separation models carry their authors' own licenses, some
  of them non-commercial. Check the licenses before using the output commercially.
