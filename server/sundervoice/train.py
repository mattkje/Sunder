"""Fine-tunes RVC v2's pretrained 40 kHz generator/discriminator on one voice
with RVC's own objective (`train/train.py`): mel L1 x45 + KL + feature matching
+ LSGAN, AdamW(1e-4, betas 0.8/0.99, eps 1e-9), lr x0.999875 per epoch,
length-bucketed batches of 4. Same recipe as the app's VoiceKit trainer."""

import random
from pathlib import Path

import torch
from torch.nn import functional as F

from rvc import commons
from rvc.models import MultiPeriodDiscriminatorV2, SynthesizerTrnMs768NSFsid
from rvc.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch

from . import assets
from .dataset import HOP, SAMPLE_RATE, SEGMENT_FRAMES, Item

CONFIG = dict(
    spec_channels=1025, segment_size=SEGMENT_FRAMES, inter_channels=192, hidden_channels=192,
    filter_channels=768, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0,
    resblock="1", resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5]] * 3, upsample_rates=[10, 10, 2, 2],
    upsample_initial_channel=512, upsample_kernel_sizes=[16, 16, 4, 4],
    spk_embed_dim=109, gin_channels=256, sr=SAMPLE_RATE, is_half=False,
)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
LR_DECAY = 0.999875
CHECKPOINT_INTERVAL = 5  # epochs


def suggested_epochs(seconds: float) -> int:
    """~8k optimizer steps whatever the dataset size, clamped to 50...300
    (the app's `VoiceWorkspace.suggestedEpochs`)."""
    if seconds <= 0:
        return 200
    return min(300, max(50, round(8000 * 4 * 3.4 / seconds)))


class Stopped(Exception):
    """Training was asked to stop; a checkpoint was saved first."""


class Trainer:
    def __init__(self, items: list[Item], device: torch.device):
        self.items, self.device = items, device
        self.net_g = SynthesizerTrnMs768NSFsid(**CONFIG).to(device)
        self.net_d = MultiPeriodDiscriminatorV2().to(device)
        self.net_g.load_state_dict(_weights(assets.base_generator()))
        self.net_d.load_state_dict(_weights(assets.base_discriminator()))
        self.opt_g = torch.optim.AdamW(self.net_g.parameters(), LEARNING_RATE, betas=(0.8, 0.99), eps=1e-9)
        self.opt_d = torch.optim.AdamW(self.net_d.parameters(), LEARNING_RATE, betas=(0.8, 0.99), eps=1e-9)
        self.completed_epochs = 0

    @property
    def steps_per_epoch(self) -> int:
        return (len(self.items) + BATCH_SIZE - 1) // BATCH_SIZE

    def save_checkpoint(self, path: Path):
        tmp = path.with_suffix(".tmp")
        torch.save({
            "g": self.net_g.state_dict(), "d": self.net_d.state_dict(),
            "opt_g": self.opt_g.state_dict(), "opt_d": self.opt_d.state_dict(),
            "epoch": self.completed_epochs,
        }, tmp)
        tmp.replace(path)

    def load_checkpoint(self, path: Path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.net_g.load_state_dict(state["g"])
        self.net_d.load_state_dict(state["d"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.completed_epochs = state["epoch"]

    def train(self, epochs: int, checkpoint: Path, should_stop, on_step):
        """Trains up to `epochs`, checkpointing every few epochs. Raises
        `Stopped` (after checkpointing) when `should_stop()` turns true."""
        self.net_g.train()
        self.net_d.train()
        while self.completed_epochs < epochs:
            epoch = self.completed_epochs + 1
            lr = LEARNING_RATE * LR_DECAY ** (epoch - 1)
            for opt in (self.opt_g, self.opt_d):
                for group in opt.param_groups:
                    group["lr"] = lr
            batches = self._batches()
            for step, batch in enumerate(batches):
                if should_stop():
                    self.save_checkpoint(checkpoint)
                    raise Stopped
                mel = self._step(batch)
                on_step(epoch, epochs, step + 1, len(batches), mel)
            if not all(torch.isfinite(p).all() for p in self.net_g.parameters()):
                raise RuntimeError("Training became numerically unstable.")
            self.completed_epochs = epoch
            if epoch % CHECKPOINT_INTERVAL == 0 or epoch == epochs:
                self.save_checkpoint(checkpoint)

    def _batches(self) -> list[list[int]]:
        """Consecutive items of similar length, in shuffled batch order."""
        order = sorted(range(len(self.items)), key=lambda i: self.items[i].frames)
        batches = [order[i : i + BATCH_SIZE] for i in range(0, len(order), BATCH_SIZE)]
        random.shuffle(batches)
        return batches

    def _step(self, indices: list[int]) -> float:
        items = [self.items[i] for i in indices]
        lengths = torch.tensor([it.frames for it in items], device=self.device)
        frames = int(lengths.max())

        def pad(arrays, scale=1):
            """Zero-pads each item's first axis to the batch's longest."""
            return torch.stack([
                F.pad(torch.from_numpy(a), [0, 0] * (a.ndim - 1) + [0, frames * scale - len(a)]) for a in arrays
            ]).to(self.device)

        phone = pad([it.content for it in items]).float()
        pitch = pad([it.pitch for it in items])
        pitchf = pad([it.pitchf for it in items])
        wave = pad([it.wave for it in items], scale=HOP)
        spec = spectrogram_torch(wave, 2048, SAMPLE_RATE, HOP, 2048, center=False)[..., :frames]
        sid = torch.zeros(len(items), dtype=torch.long, device=self.device)

        y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = self.net_g(phone, lengths, pitch, pitchf, spec, lengths, sid)
        mel = spec_to_mel_torch(spec, 2048, 125, SAMPLE_RATE, 0.0, None)
        y_mel = commons.slice_segments(mel, ids_slice, SEGMENT_FRAMES)
        y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), 2048, 125, SAMPLE_RATE, HOP, 2048, 0.0, None)
        real = commons.slice_segments(wave.unsqueeze(1), ids_slice * HOP, SEGMENT_FRAMES * HOP)

        d_real, d_fake, _, _ = self.net_d(real, y_hat.detach())
        _guarded_step(self.opt_d, self.net_d, _discriminator_loss(d_real, d_fake))

        d_real, d_fake, fmap_real, fmap_fake = self.net_d(real, y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
        loss_kl = _kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        loss = loss_mel + loss_kl + _feature_loss(fmap_real, fmap_fake) + _generator_loss(d_fake)
        _guarded_step(self.opt_g, self.net_g, loss)
        return float(loss_mel.detach())


def _weights(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


def _guarded_step(opt, model, loss):
    """Skips the update if the loss or any gradient isn't finite -- what RVC's
    fp16 GradScaler does implicitly; one bad batch would otherwise poison every
    weight with NaN."""
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads):
        opt.step()
    opt.zero_grad(set_to_none=True)


def _feature_loss(fmap_r, fmap_g):
    return 2 * sum(torch.mean(torch.abs(r.float().detach() - g.float())) for dr, dg in zip(fmap_r, fmap_g) for r, g in zip(dr, dg))


def _discriminator_loss(real, fake):
    return sum(torch.mean((1 - r.float()) ** 2) + torch.mean(g.float() ** 2) for r, g in zip(real, fake))


def _generator_loss(fake):
    return sum(torch.mean((1 - g.float()) ** 2) for g in fake)


def _kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)
