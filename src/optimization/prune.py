from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.prune as prune


def apply_magnitude_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def iterative_prune_and_finetune(
    model: nn.Module,
    trainer,
    amounts: list[float] = (0.2, 0.3, 0.5),
    finetune_epochs: int = 3,
) -> nn.Module:
    import copy
    original_epochs = trainer.config.epochs
    trainer.config.epochs = finetune_epochs

    best_model = copy.deepcopy(model)
    best_auc = 0.0

    for amount in amounts:
        apply_magnitude_pruning(model, amount)
        trainer.model = model
        trainer.fit()

        _, _, val_auc = trainer.val_epoch()
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = copy.deepcopy(model)

    trainer.config.epochs = original_epochs
    return best_model
