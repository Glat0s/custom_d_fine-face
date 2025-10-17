from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from src.d_fine.dfine_criterion import DFINECriterion
from src.d_fine.deim_criterion import DEIMCriterion

from .arch.dfine_decoder import DFINETransformer
from .arch.hgnetv2 import HGNetv2
from .arch.hybrid_encoder import HybridEncoder
from .configs import models
from .matcher import HungarianMatcher
from .utils import load_tuning_state

__all__ = ["DFINE"]


class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


def build_model(model_name, num_classes, device, img_size=None, pretrained_model_path=None):
    model_cfg = models[model_name]
    model_cfg["HybridEncoder"]["eval_spatial_size"] = img_size
    model_cfg["DFINETransformer"]["eval_spatial_size"] = img_size

    backbone = HGNetv2(**model_cfg["HGNetv2"])
    encoder = HybridEncoder(**model_cfg["HybridEncoder"])
    decoder = DFINETransformer(num_classes=num_classes, **model_cfg["DFINETransformer"])

    model = DFINE(backbone, encoder, decoder)

    if pretrained_model_path:
        if not Path(pretrained_model_path).exists():
            raise FileNotFoundError(f"{pretrained_model_path} does not exist")
        model = load_tuning_state(model, str(pretrained_model_path))
    return model.to(device)

def build_loss(cfg, num_classes): # <-- Changed signature to pass full cfg
    """Builds the loss function based on the model_type in the config."""
    model_cfg = models[cfg.model_name]
    matcher = HungarianMatcher(**model_cfg["matcher"])

    if cfg.model_type == 'deim':
        print("Building DEIM Loss (DEIMCriterion)...")
        # Make a copy to avoid modifying the original config
        dfine_criterion_cfg = model_cfg["DFINECriterion"].copy()
        # Remove weight_dict and losses to avoid conflict
        del dfine_criterion_cfg["weight_dict"]
        del dfine_criterion_cfg["losses"]
        del dfine_criterion_cfg["gamma"]

        loss_fn = DEIMCriterion(
            matcher,
            num_classes=num_classes,
            # DEIM uses MAL loss instead of VFL
            losses=['mal', 'boxes', 'local'], 
            weight_dict={
                "loss_mal": 1,
                "loss_bbox": 5,
                "loss_giou": 2,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            gamma=cfg.train.deim.mal_gamma, # Use DEIM specific gamma
            use_uni_set=cfg.train.deim.use_uni_set,
            **dfine_criterion_cfg # Re-use other params
        )
    else: # Default to original D-FINE
        print("Building D-FINE Loss (DFINECriterion)...")
        loss_fn = DFINECriterion(
            matcher,
            num_classes=num_classes,
            label_smoothing=cfg.train.label_smoothing,
            **model_cfg["DFINECriterion"],
        )

    return loss_fn

def build_optimizer(model, lr, backbone_lr, betas, weight_decay, base_lr):
    backbone_exclude_norm = []
    backbone_norm = []
    encdec_norm_bias = []
    rest = []

    for name, param in model.named_parameters():
        # Group 1 and 2: "backbone" in name
        if "backbone" in name:
            if "norm" in name or "bn" in name:
                # Group 2: backbone + norm/bn
                backbone_norm.append(param)
            else:
                # Group 1: backbone but not norm/bn
                backbone_exclude_norm.append(param)

        # Group 3: "encoder" or "decoder" plus "norm"/"bn"/"bias"
        elif ("encoder" in name or "decoder" in name) and (
            "norm" in name or "bn" in name or "bias" in name
        ):
            encdec_norm_bias.append(param)

        else:
            rest.append(param)

    group1 = {"params": backbone_exclude_norm, "lr": backbone_lr, "initial_lr": backbone_lr}
    group2 = {
        "params": backbone_norm,
        "lr": backbone_lr,
        "weight_decay": 0.0,
        "initial_lr": backbone_lr,
    }
    group3 = {"params": encdec_norm_bias, "weight_decay": 0.0, "lr": base_lr, "initial_lr": base_lr}
    group4 = {"params": rest, "lr": base_lr, "initial_lr": base_lr}

    param_groups = [group1, group2, group3, group4]

    return optim.AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)
