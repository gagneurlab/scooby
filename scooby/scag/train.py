"""Training entry point for scAG.

Usage::

    accelerate launch --config_file scripts/scag/accelerate_config.yaml \
        -m scooby.scag.train --config scooby/scag/configs/neurips.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import TorchDynamoPlugin
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader

from scooby.utils.utils import evaluate, poisson_multinomial_torch

from .config import ScAGConfig, TrainingConfig
from .data import build_datasets
from .lora import StandardizedConvLoRA, apply_lora, trainable_parameter_summary
from .model import ScoobyAlphaGenome
from .predict import fix_rev_comp

# An output this small means softplus is saturated: its gradient has
# underflowed and no parameter can move again.
COLLAPSE_OUTPUT_THRESHOLD = 1e-8


class CollapseGuard:
    """Stop training when the output head saturates.

    The decoder ends in a softplus; if the sequence embedding drifts, the
    output underflows to zero and the gradient vanishes. Training stops when
    the median of each step's largest prediction over a window falls below
    ``threshold``. :meth:`check` synchronises the GPU, so call it only every
    few steps.

    Args:
        patience: Window of recent steps whose median is compared with the
            threshold. ``0`` disables the check.
        threshold: Median largest prediction below which training is
            considered collapsed.
    """

    def __init__(self, patience: int, threshold: float = COLLAPSE_OUTPUT_THRESHOLD):
        self.patience = patience
        self.threshold = threshold
        self.peaks: Optional[torch.Tensor] = None
        self.recorded = 0

    def record(self, outputs: torch.Tensor) -> None:
        """Store this step's largest prediction, without leaving the GPU."""
        if self.patience <= 0:
            return
        if self.peaks is None:
            self.peaks = torch.empty(self.patience, device=outputs.device)
        self.peaks[self.recorded % self.patience] = outputs.detach().float().max()
        self.recorded += 1

    def check(self, step: int) -> Optional[str]:
        """Return a diagnostic message when the guard trips, else ``None``."""
        if self.patience <= 0 or self.recorded < self.patience:
            return None
        median = self.peaks.median().item()
        if median >= self.threshold:
            return None
        return (
            f"output collapsed: the median largest prediction over the last "
            f"{self.patience} steps is {median:.3g}, below {self.threshold:g} "
            f"(step {step}). The softplus in the decoder is saturated and "
            "gradients have vanished, so training cannot recover. Lower "
            "training.lr or training.encoder_lora_lr, or set "
            "model.head_input_norm: true."
        )


class StepMetrics:
    """Accumulate per-step training metrics on the GPU; read them back rarely.

    Logging a tensor every step forces a device synchronisation. This keeps
    the loss sum and the per-step output peaks on the device and only
    transfers them in :meth:`flush`.
    """

    def __init__(self):
        self.loss_sum: Optional[torch.Tensor] = None
        self.peaks: list[torch.Tensor] = []

    def update(self, loss: torch.Tensor, outputs: torch.Tensor) -> None:
        loss = loss.detach().float()
        self.loss_sum = loss if self.loss_sum is None else self.loss_sum + loss
        self.peaks.append(outputs.detach().float().max())

    def flush(self) -> dict:
        """Mean loss and median per-step output peak since the last flush."""
        if not self.peaks:
            return {}
        stats = {
            "loss": (self.loss_sum / len(self.peaks)).item(),
            "output_max": torch.stack(self.peaks).median().item(),
        }
        self.loss_sum, self.peaks = None, []
        return stats


def parameter_groups(model: nn.Module, cfg: TrainingConfig):
    """Split parameters by weight decay and learning rate.

    The cell-state decoder and the convolutional adapters get their own
    learning rates, since both are more sensitive than the transformer
    adapters.
    """
    conv_lora_params = set()
    for module in model.modules():
        if isinstance(module, StandardizedConvLoRA):
            conv_lora_params.update(id(p) for p in module.parameters(recurse=False))

    decay, no_decay, head, conv_lora = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in conv_lora_params:
            conv_lora.append(param)
        elif "cell_state_to_conv" in name:
            head.append(param)
        elif param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": head, "weight_decay": cfg.wd, "lr": cfg.cellstate_lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": cfg.lr},
        {"params": decay, "weight_decay": cfg.wd, "lr": cfg.lr},
    ]
    if conv_lora:
        groups.append({
            "params": conv_lora,
            "weight_decay": cfg.wd,
            "lr": cfg.encoder_lora_lr if cfg.encoder_lora_lr is not None else cfg.lr,
        })
    return [g for g in groups if g["params"]]


def dynamo_plugin(cfg: TrainingConfig) -> TorchDynamoPlugin:
    """Build the compile plugin, or a disabled one when compiling is off."""
    if not cfg.compile:
        return TorchDynamoPlugin(backend="no")
    if cfg.compile_mode == "reduce-overhead":
        print(
            "warning: training.compile_mode 'reduce-overhead' uses CUDA "
            "graphs, which do not combine with DDP. Use 'default' or "
            "'max-autotune-no-cudagraphs' for a multi-GPU run."
        )
    return TorchDynamoPlugin(backend="inductor", mode=cfg.compile_mode)


def build_model(config: ScAGConfig, cell_emb_dim: Optional[int] = None):
    """Construct the model described by ``config``.

    Args:
        config: The run config.
        cell_emb_dim: Width found in the embedding table. When it disagrees
            with the config, the config wins and a warning is printed, since
            a mismatch usually means the wrong embedding file.
    """
    model_cfg = config.model
    if cell_emb_dim is not None and cell_emb_dim != model_cfg.cell_emb_dim:
        print(
            f"warning: model.cell_emb_dim is {model_cfg.cell_emb_dim} but the "
            f"embedding table has width {cell_emb_dim}"
        )

    model = ScoobyAlphaGenome(
        ag_pretrained_path=model_cfg.pretrained_model,
        disable_cache=True,
        **model_cfg.model_kwargs(),
    )
    if model_cfg.use_lora:
        model = apply_lora(model, model_cfg, train=True)
    else:
        for p in model.parameters():
            p.requires_grad = True
        if model_cfg.freeze_batchnorm:
            model._freeze_norm_layers()
    return model


def train(config: ScAGConfig) -> None:
    """Run training as described by ``config``."""
    tcfg = config.training
    accelerator = Accelerator(
        log_with="wandb",
        kwargs_handlers=[
            DistributedDataParallelKwargs(static_graph=tcfg.ddp_static_graph)
        ],
        step_scheduler_with_optimizer=False,
        # Defer to `accelerate launch`; bf16 when run with plain python.
        mixed_precision=os.environ.get("ACCELERATE_MIXED_PRECISION", "bf16"),
        # accelerate compiles in `prepare`, after wrapping the model in DDP.
        dynamo_plugin=dynamo_plugin(tcfg),
    )
    device = accelerator.device

    data_cfg = config.require_data()
    n_tracks = data_cfg.num_tracks
    eval_mode = "rna" if data_cfg.rna_only else "multiome"
    train_dataset, val_dataset, cell_emb_dim = build_datasets(data_cfg)
    model = build_model(config, cell_emb_dim)
    trainable, total = trainable_parameter_summary(model)
    accelerator.print(
        f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)"
    )
    if tcfg.compile:
        accelerator.print(
            f"compiling with TorchInductor (mode={tcfg.compile_mode}); the "
            "first steps include roughly 100 s of warm-up"
        )

    # Each optimizer step consumes batch_size * num_processes sequences.
    steps_per_epoch = tcfg.steps_per_epoch // (
        tcfg.batch_size * accelerator.num_processes
    )
    num_steps = steps_per_epoch * tcfg.num_epochs
    accelerator.print(
        f"{accelerator.num_processes} process(es), {steps_per_epoch} optimizer "
        f"steps per epoch, {num_steps} total"
    )
    if tcfg.warmup_steps >= num_steps:
        raise ValueError(
            f"training.warmup_steps ({tcfg.warmup_steps}) is at least the "
            f"total number of optimizer steps ({num_steps}) on "
            f"{accelerator.num_processes} process(es), so the learning rate "
            "would never leave warm-up. Lower warmup_steps or raise "
            "num_epochs."
        )

    optimizer = torch.optim.AdamW(parameter_groups(model, tcfg))
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=1e-4, total_iters=tcfg.warmup_steps),
            LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_steps - tcfg.warmup_steps,
            ),
        ],
        [tcfg.warmup_steps],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=tcfg.batch_size,
        shuffle=True,
        num_workers=tcfg.num_workers,
        persistent_workers=tcfg.num_workers > 0,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        if val_dataset is not None
        else None
    )
    if val_loader is None:
        accelerator.print("no validation sequences; skipping evaluation")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    prepared = accelerator.prepare(model, optimizer, scheduler, train_loader)
    model, optimizer, scheduler, train_loader = prepared
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    accelerator.init_trackers("scooby", init_kwargs={"wandb": {"name": config.run_name}})
    guard = CollapseGuard(tcfg.collapse_patience, tcfg.collapse_threshold)
    metrics = StepMetrics()
    global_step = 0

    for epoch in range(tcfg.num_epochs):
        for step, (inputs, rc_augs, targets, cell_emb_idx) in tqdm.tqdm(
            enumerate(train_loader), disable=not accelerator.is_main_process
        ):
            # AlphaGenome takes NLC (B, S, 4), which is what the loader
            # yields, so no permutation here.
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            for idx in rc_augs.nonzero():
                idx = idx[0]
                flipped = torch.flip(targets[idx].unsqueeze(0), (1, -3))
                targets[idx] = fix_rev_comp(flipped, n_tracks)[0]

            optimizer.zero_grad()
            with accelerator.autocast():
                outputs = model(inputs, cell_emb_idx)
                loss = poisson_multinomial_torch(
                    outputs, targets, total_weight=tcfg.total_weight,
                    epsilon=tcfg.loss_epsilon,
                )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), tcfg.clip_global_norm)
            optimizer.step()
            scheduler.step()

            # Per-step values stay on the GPU; reading them back every step
            # would synchronise the device each step.
            metrics.update(loss, outputs)
            guard.record(outputs)
            global_step += 1
            if global_step % tcfg.log_every_n == 0:
                # Log against the training step, not wandb's call counter.
                accelerator.log({
                    **metrics.flush(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "train_step": global_step,
                }, step=global_step)
                message = guard.check(global_step)
                if message is not None:
                    accelerator.print(message)
                    raise RuntimeError(message)

            if val_loader is not None and step % tcfg.eval_every_n == 0:
                evaluate(accelerator, model, val_loader, mode=eval_mode)
                model.train()
            if step % tcfg.save_every_n == 0 and (step or epoch):
                save_checkpoint(accelerator, model, config, f"epoch_{epoch}_step_{step}")

    save_checkpoint(accelerator, model, config, "final")
    accelerator.end_training()


def save_checkpoint(accelerator, model, config: ScAGConfig, tag: str) -> str:
    """Write a checkpoint that :meth:`ScoobyAlphaGenome.from_pretrained` can read."""
    path = os.path.join(config.checkpoint_root, tag)
    if accelerator.is_main_process:
        os.makedirs(path, exist_ok=True)
    accelerator.wait_for_everyone()
    # keep_torch_compile=False strips the `_orig_mod.` prefix that compiling
    # adds, so checkpoints are identical whether or not the run compiled.
    unwrapped = accelerator.unwrap_model(model, keep_torch_compile=False)
    if accelerator.is_main_process:
        config.dump(os.path.join(path, "scag_config.yaml"))
        from safetensors.torch import save_file

        state = {k: v.contiguous().cpu() for k, v in unwrapped.state_dict().items()}
        save_file(state, os.path.join(path, "model.safetensors"))
    accelerator.save_state(output_dir=path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scAG")
    parser.add_argument("--config", required=True, help="path to a scAG YAML config")
    parser.add_argument(
        "--compile", dest="compile", action="store_true", default=None,
        help="override training.compile to true",
    )
    parser.add_argument(
        "--no-compile", dest="compile", action="store_false",
        help="override training.compile to false",
    )
    args = parser.parse_args()

    config = ScAGConfig.from_yaml(args.config)
    if args.compile is not None:
        config.training.compile = args.compile
    train(config)


if __name__ == "__main__":
    main()
