"""Seed-VC fine-tuning for one voice: the singing (f0-conditioned, 44.1 kHz,
BigVGAN) model only. Same objective as upstream `train.py`: the source is
timbre-perturbed through OpenVoice so the model learns to take timbre from
the prompt, a random prefix of each clip serves as that prompt, and the CFM
flow-matching loss trains `cfm` + `length_regulator`. Upstream's CLI,
alternative tokenizers/vocoders and directory-scanning checkpoint logic are
dropped; the caller owns the loop, checkpoints and stopping."""

import os

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

from seedvc.commons import recursive_munch, build_model, load_checkpoint
from seedvc.optimizers import build_optimizer
from seedvc.ft_dataset import build_ft_dataloader
from seedvc.hf_utils import load_custom_model_from_hf

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_f0_44k.yml")
# Fine-tuning starts from the same weights zero-shot inference uses (upstream
# trains from the older `..._ft_ema.pth`).
BASE_MODEL = "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
LEARNING_RATE = 1e-5


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_config():
    return yaml.safe_load(open(CONFIG_PATH))


def build_dit(config, device):
    params = recursive_munch(config["model_params"])
    params.dit_type = "DiT"
    model = build_model(params, stage="DiT")
    for key in model:
        model[key].to(device)
    return model


def load_dit(checkpoint: str | None, device="cpu"):
    """The DiT (`cfm` + `length_regulator`) from a fine-tuned checkpoint, or
    the pretrained base when `checkpoint` is None, in eval mode."""
    model = build_dit(load_config(), device)
    path = checkpoint or load_custom_model_from_hf("Plachta/Seed-VC", BASE_MODEL, None)
    model, *_ = load_checkpoint(model, None, path, load_only_params=True, ignore_modules=[], is_distributed=False)
    for key in model:
        model[key].eval()
    model.cfm.estimator.setup_caches(max_batch_size=2, max_seq_length=8192)
    return model


class Trainer:
    def __init__(self, data_dir: str, device: str, batch_size: int = 2):
        self.device = device
        config = load_config()
        pre = config["preprocess_params"]
        self.sr = pre["sr"]
        self.dataloader = build_ft_dataloader(data_dir, pre["spect_params"], self.sr, batch_size=batch_size)

        from seedvc.campplus.DTDNN import CAMPPlus
        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus.load_state_dict(torch.load(
            load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", None), map_location="cpu"))
        self.campplus.eval().to(device)

        from seedvc.rmvpe import RMVPE
        self.rmvpe = RMVPE(load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None),
                           is_half=False, device=device)

        from seedvc.openvoice.api import ToneColorConverter
        ckpt, cfg = load_custom_model_from_hf("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", "converter/config.json")
        self.tone_color_converter = ToneColorConverter(cfg, device=device)
        self.tone_color_converter.load_ckpt(ckpt)
        self.tone_color_converter.model.eval()
        self.se_db = torch.load(load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt", None), map_location="cpu")

        from transformers import AutoFeatureExtractor, WhisperModel
        name = config["model_params"]["speech_tokenizer"]["name"]
        self.whisper = WhisperModel.from_pretrained(name).to(device)
        del self.whisper.decoder
        self.whisper_features = AutoFeatureExtractor.from_pretrained(name)

        self.model = build_dit(config, device)
        self.model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)
        self.optimizer = build_optimizer({key: self.model[key] for key in self.model}, lr=LEARNING_RATE)
        base = load_custom_model_from_hf("Plachta/Seed-VC", BASE_MODEL, None)
        self.model, self.optimizer, _, _ = load_checkpoint(
            self.model, self.optimizer, base, load_only_params=True, ignore_modules=[], is_distributed=False)
        self.steps = 0

    @property
    def steps_per_epoch(self) -> int:
        return len(self.dataloader)

    def save_checkpoint(self, path: str):
        tmp = path + ".tmp"
        torch.save({
            "net": {key: self.model[key].state_dict() for key in self.model},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.optimizer.scheduler_state_dict(),
            "iters": self.steps, "epoch": 0,
        }, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location="cpu")
        for key in self.model:
            self.model[key].load_state_dict(state["net"][key])
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer.load_scheduler_state_dict(state["scheduler"])
        self.steps = state["iters"]

    def save_weights(self, path: str):
        """Weights only (no optimizer), the input to the voice export."""
        torch.save({"net": {key: self.model[key].state_dict() for key in self.model}}, path)

    def train(self, steps: int, on_step, should_stop):
        """Trains until `self.steps == steps` or `should_stop()`; returns True when done."""
        for key in self.model:
            self.model[key].train()
        while self.steps < steps:
            for batch in self.dataloader:
                if should_stop():
                    return False
                loss = self.train_step([b.to(self.device) for b in batch])
                self.steps += 1
                # Every batch has a different length, so PyTorch's allocator
                # cache keeps growing; on a Mac it pushes training into swap
                # (4 s/step became 16 s/step within ~80 steps).
                if self.device == "mps" and self.steps % 10 == 0:
                    torch.mps.empty_cache()
                on_step(self.steps, loss)
                if self.steps >= steps:
                    break
        return True

    def _semantic(self, waves_16k):
        inputs = self.whisper_features([w.cpu().numpy() for w in waves_16k], return_tensors="pt",
                                       return_attention_mask=True, sampling_rate=16000)
        feats = self.whisper._mask_input_features(inputs.input_features, attention_mask=inputs.attention_mask)
        with torch.no_grad():
            out = self.whisper.encoder(feats.to(self.device)).last_hidden_state.float()
        return out[:, :waves_16k.size(-1) // 320 + 1]

    def train_step(self, batch) -> float:
        waves, mels, wave_lengths, mel_lengths = batch
        B = waves.size(0)

        # Timbre-perturbed copy of the source: its content tokens become the
        # condition, so the model can't copy timbre from them.
        waves_22k = torchaudio.functional.resample(waves, self.sr, 22050)
        lengths_22k = (wave_lengths.float() * 22050 / self.sr).long()
        with torch.no_grad():
            se = self.tone_color_converter.extract_se(waves_22k, lengths_22k)
            target_se = self.se_db[torch.randint(0, len(self.se_db), (B,))].to(self.device)
            converted_22k = self.tone_color_converter.convert(waves_22k, lengths_22k, se, target_se).squeeze(1)
        converted = torchaudio.functional.resample(converted_22k, 22050, self.sr)

        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        lengths_16k = (wave_lengths.float() * 16000 / self.sr).long()
        converted_16k = torchaudio.functional.resample(converted, self.sr, 16000)
        S_ori, S_alt = self._semantic(waves_16k), self._semantic(converted_16k)
        F0_ori = self.rmvpe.infer_from_audio_batch(waves_16k)

        alt_cond = self.model.length_regulator(S_alt, ylens=mel_lengths, f0=F0_ori)[0]
        ori_cond = self.model.length_regulator(S_ori, ylens=mel_lengths, f0=F0_ori)[0]

        # A random prefix is the prompt (10% of the time none); it uses the
        # unperturbed tokens, as at inference.
        prompt_len = (torch.rand([B], device=self.device) * (mel_lengths - 1)).floor().long()
        prompt_len[torch.rand([B], device=self.device) < 0.1] = 0
        cond = alt_cond.clone()
        for b in range(B):
            cond[b, :prompt_len[b]] = ori_cond[b, :prompt_len[b]]

        common = min(mels.size(2), cond.size(1))
        x, cond = mels[:, :, :common], cond[:, :common]
        mel_lengths = torch.clamp(mel_lengths, max=common)

        styles = []
        with torch.no_grad():
            for b in range(B):
                feat = kaldi.fbank(waves_16k[b:b + 1, :lengths_16k[b]], num_mel_bins=80, dither=0, sample_frequency=16000)
                styles.append(self.campplus((feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0)))
        style = torch.cat(styles, dim=0)

        loss, _ = self.model.cfm(x, mel_lengths, prompt_len, cond, style)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.cfm.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.length_regulator.parameters(), 10.0)
        for key in ("cfm", "length_regulator"):
            self.optimizer.step(key)
            self.optimizer.scheduler(key=key)
        return loss.detach().item()
