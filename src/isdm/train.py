from typing import Optional, Dict, Any
import torch
from torch import nn
from torch.utils.data import DataLoader
import itertools


def train_single_source(
    model: nn.Module,
    criterion,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_cls=torch.optim.Adam,
    wandb_run=None,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, list]:
    """
    Generic trainer for one-source supervised training.

    Source-specific behavior should live in:
    - the dataset / dataloader
    - the criterion
    - the model
    - the W&B config, not inside this function
    """

    model.to(device)
    print(f"Training on device: {device}")

    if isinstance(criterion, nn.Module):
        criterion.to(device)

    optimizer = optimizer_cls(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    global_step = 0

    for epoch in range(1, epochs + 1):
        # ------------------
        # Train
        # ------------------
        model.train()
        running_train = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )

            optimizer.step()

            batch_size = xb.size(0)
            running_train += loss.item() * batch_size
            n_train += batch_size
            global_step += 1

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        train_loss = running_train / max(n_train, 1)

        # ------------------
        # Validation
        # ------------------
        val_loss = None

        if val_loader is not None:
            model.eval()
            running_val = 0.0
            n_val = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    logits = model(xb)
                    loss = criterion(logits, yb)

                    batch_size = xb.size(0)
                    running_val += loss.item() * batch_size
                    n_val += batch_size

            val_loss = running_val / max(n_val, 1)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # ------------------
        # Epoch logging
        # ------------------
        epoch_log = {
            "train/loss": train_loss,
            "train/lr_epoch": current_lr,
            "epoch": epoch,
        }

        if val_loss is not None:
            epoch_log["val/loss"] = val_loss

        if wandb_run is not None:
            wandb_run.log(epoch_log, step=global_step)

        if val_loss is None:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f}"
            )

    return history






def train_double_source(
    model: nn.Module,
    criterion: nn.Module,
    source1_loader: DataLoader,
    source2_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_cls=torch.optim.Adam,
    wandb_run=None,
    source1_name: str = "source1",
    source2_name: str = "source2",
    grad_clip_norm: Optional[float] = None,
    cycle_shorter_loader: bool = True,
    max_steps_per_epoch: Optional[int] = None
) -> Dict[str, list]:

    model.to(device)
    print(f"Training on device: {device}")

    if isinstance(criterion, nn.Module):
        criterion.to(device)

    optimizer = optimizer_cls(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        f"{source1_name}_loss": [],
        f"{source2_name}_loss": [],
    }

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()

        running_total = 0.0
        running_s1 = 0.0
        running_s2 = 0.0
        n_steps = 0

        if cycle_shorter_loader:
            max_len = max(len(source1_loader), len(source2_loader))

            if len(source1_loader) >= len(source2_loader):
                iterator = zip(
                    source1_loader,
                    itertools.cycle(source2_loader),
                )
            else:
                iterator = zip(
                    itertools.cycle(source1_loader),
                    source2_loader,
                )

            iterator = itertools.islice(iterator, max_len)

        else:
            iterator = zip(source1_loader, source2_loader)
        
        # Cap steps regardless of which branch was taken
        if max_steps_per_epoch is not None:
            iterator = itertools.islice(iterator, max_steps_per_epoch)

        for batch1, batch2 in iterator:
            xb1, yb1 = batch1
            xb2, yb2 = batch2

            xb1 = xb1.to(device, non_blocking=True)
            yb1 = yb1.to(device, non_blocking=True)
            xb2 = xb2.to(device, non_blocking=True)
            yb2 = yb2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred1 = model(xb1)
            pred2 = model(xb2)

            loss, loss_logs = criterion(
                pred_source1=pred1,
                pred_source2=pred2,
                target_source1=yb1,
                target_source2=yb2,
            )

            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )

            optimizer.step()

            global_step += 1
            n_steps += 1

            loss_value = loss.item()
            source1_loss = loss_logs["loss/source1"].item()
            source2_loss = loss_logs["loss/source2"].item()

            running_total += loss_value
            running_s1 += source1_loss
            running_s2 += source2_loss

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/batch_loss": loss_value,
                        f"train/{source1_name}_batch_loss": source1_loss,
                        f"train/{source2_name}_batch_loss": source2_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        epoch_total = running_total / max(n_steps, 1)
        epoch_s1 = running_s1 / max(n_steps, 1)
        epoch_s2 = running_s2 / max(n_steps, 1)

        history["train_loss"].append(epoch_total)
        history[f"{source1_name}_loss"].append(epoch_s1)
        history[f"{source2_name}_loss"].append(epoch_s2)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": epoch_total,
                    f"train/{source1_name}_loss": epoch_s1,
                    f"train/{source2_name}_loss": epoch_s2,
                    "epoch": epoch,
                },
                step=global_step,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={epoch_total:.4f} | "
            f"{source1_name}={epoch_s1:.4f} | "
            f"{source2_name}={epoch_s2:.4f}"
        )

    return history