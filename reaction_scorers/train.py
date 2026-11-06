"""Train RP or RGP from preprocessed reaction CSVs.

Model configs specify the architecture and graph vocabulary.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .graph.featurizer import ReactionGraphFeaturizer
from .loading import build_rgp
from .rp_collate import collate_rp
from .rp_dataset import RPTrainingDataset
from .rp_loss import token_loss
from .schedules import update_lr_for_step, warmup_and_cosine_decay
from .training_data import RGPTrainingDataset, collate_rgp, graph_sample_weights
from .transformer import ReactionTransformer

MONITORS = {"rp": "val/TokenAccuracy", "rgp": "val/BinaryAUROC"}


class TrainingModule(pl.LightningModule):
    """Shared training loop; RP predicts tokens, RGP predicts reaction labels."""

    def __init__(self, model, kind):
        super().__init__()
        self.model, self.kind = model, kind
        self.validation_values = []

    def forward(self, batch):
        return self.model(batch)

    def loss(self, batch):
        logits = self(batch["x"])
        if self.kind == "rp":
            return token_loss(logits, batch["y"]), logits
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["y"], reduction="none"
        )
        return (losses * batch["weight"]).mean(), logits

    def training_step(self, batch, batch_idx):
        loss, _ = self.loss(batch)
        self.log("train/Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.loss(batch)
        self.log("val/Loss", loss, batch_size=len(batch["y"]))
        if self.kind == "rp":
            mask = batch["y"] != 0
            correct = ((logits.argmax(-1) == batch["y"]) & mask).sum()
            self.validation_values.append((int(correct), int(mask.sum())))
        else:
            self.validation_values.append(
                (batch["y"].detach().cpu(), torch.sigmoid(logits).detach().cpu())
            )

    def on_validation_epoch_end(self):
        if self.kind == "rp":
            correct, total = map(sum, zip(*self.validation_values))
            metric = correct / total if total else 0.0
        else:
            labels = torch.cat([values[0] for values in self.validation_values]).numpy()
            scores = torch.cat([values[1] for values in self.validation_values]).numpy()
            metric = roc_auc_score(labels, scores)
        self.log(MONITORS[self.kind], float(metric))
        self.validation_values.clear()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        if self.kind == "rgp":
            update_lr_for_step(optimizer, self.global_step)
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        if self.kind == "rp":
            optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=0.05)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, warmup_and_cosine_decay
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # The training settings minimize AUROC for LR scheduling and maximize
        # it for checkpoint selection.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": MONITORS["rgp"],
        }


def training_inputs(kind, train, validation, models_dir):
    """Construct the saved architecture and per-split datasets."""
    config = json.loads((models_dir / kind / "config.json").read_text())
    if kind == "rp":
        network = ReactionTransformer(config["architecture"])

        def make_dataset(frame):
            return RPTrainingDataset(frame.substrates, frame["product"])

        return network, make_dataset(train), make_dataset(validation), collate_rp
    featurizer = ReactionGraphFeaturizer(
        models_dir / "rgp/feature_values.json", **config["featurizer"]
    )
    weights = graph_sample_weights(train)
    train_data = RGPTrainingDataset(train, featurizer, weights)
    val_data = RGPTrainingDataset(validation, featurizer, weights)
    return build_rgp(config), train_data, val_data, collate_rgp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rp", "rgp"], required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    pl.seed_everything(20200101, workers=True)
    model, train, validation, collate = training_inputs(
        args.model,
        pd.read_csv(args.train_csv),
        pd.read_csv(args.val_csv),
        args.models_dir,
    )
    loader = partial(
        DataLoader,
        batch_size=8 if args.model == "rp" else 128,
        collate_fn=collate,
        num_workers=args.workers,
    )
    monitor = MONITORS[args.model]
    callbacks = [
        ModelCheckpoint(monitor=monitor, mode="max", save_last=True),
        EarlyStopping(
            monitor=monitor, mode="max", patience=300 if args.model == "rp" else 8
        ),
    ]
    trainer = pl.Trainer(
        default_root_dir=args.output,
        accelerator=args.accelerator,
        devices=1,
        precision="16-mixed",
        max_steps=495944 if args.model == "rp" else -1,
        max_epochs=-1 if args.model == "rp" else 1000,
        min_epochs=1,
        min_steps=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        gradient_clip_val=2.0 if args.model == "rp" else None,
        callbacks=callbacks,
    )
    trainer.fit(
        TrainingModule(model, args.model),
        loader(train, shuffle=True),
        loader(validation, shuffle=args.model == "rgp"),
    )


if __name__ == "__main__":
    main()
