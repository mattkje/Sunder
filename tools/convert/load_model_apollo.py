"""Loads an Apollo checkpoint into an Apollo model built from its own
config. Parametrized by a checkpoint directory (containing config.yaml and
model.{ckpt,bin}), same shape as load_model_bs.py.

Both "Apollo Universal" and "Lew Vocal Enhancer v2 beta" checkpoints store
{"model_name", "state_dict", "model_args", "infos"} with `state_dict` keys
matching Apollo's own module names directly (BN.*/net.*/output.*) -- no
prefix stripping needed, confirmed by inspection.
"""

import os
import glob
import yaml
import torch

from models.apollo.apollo import Apollo


def load_config(checkpoint_dir):
    return yaml.unsafe_load(open(os.path.join(checkpoint_dir, 'config.yaml')))


def _checkpoint_path(checkpoint_dir):
    for name in ('model.ckpt', 'model.bin'):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            return path
    matches = glob.glob(os.path.join(checkpoint_dir, 'model.*'))
    if matches:
        return matches[0]
    raise FileNotFoundError(f'no model.ckpt/model.bin in {checkpoint_dir}')


def load_model(checkpoint_dir):
    cfg = load_config(checkpoint_dir)
    m = cfg['model']
    model = Apollo(sr=m['sr'], win=m['win'], feature_dim=m['feature_dim'], layer=m['layer'])
    sd = torch.load(_checkpoint_path(checkpoint_dir), map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(sd['state_dict'], strict=True)
    assert not missing and not unexpected
    model = model.float().eval()
    return model, cfg
