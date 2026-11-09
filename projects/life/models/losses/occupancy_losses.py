import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


def inverse_sigmoid(x):
    x = x.to(torch.float32)
    while x >= 1 - 1e-5:
        x = x - 1e-5

    while x < 1e-5:
        x = x + 1e-5

    return -torch.log((1 / x) - 1)


def geo_scal_loss_with_mask(
    pred, ssc_target, camera_mask, ignore_index=255, non_empty_idx=0
):
    """Compute geometric scene-affinity loss on camera-visible tokens.

    Inputs are class logits [T, C], labels [T], and visibility [T].
    non_empty_idx identifies the empty-space class.
    """

    pred = F.softmax(pred, dim=1)

    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1 - empty_probs

    camera_mask = camera_mask.bool()

    mask = (ssc_target != ignore_index) & camera_mask
    nonempty_target = ssc_target != non_empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum() + eps)
    recall = intersection / (nonempty_target.sum() + eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (
        (1 - nonempty_target).sum() + eps
    )
    with autocast(False):
        return (
            F.binary_cross_entropy_with_logits(
                inverse_sigmoid(precision), torch.ones_like(precision)
            )
            + F.binary_cross_entropy_with_logits(
                inverse_sigmoid(recall), torch.ones_like(recall)
            )
            + F.binary_cross_entropy_with_logits(
                inverse_sigmoid(spec), torch.ones_like(spec)
            )
        )


def sem_scal_loss_with_mask(pred_, ssc_target, camera_mask, ignore_index=255):
    """Compute semantic scene-affinity loss on camera-visible tokens.

    Inputs are class logits [T, C], labels [T], and visibility [T].
    """
    with autocast(False):

        pred = F.softmax(pred_, dim=1)
        loss = 0
        count = 0
        camera_mask = camera_mask.bool()
        mask = (ssc_target != ignore_index) & camera_mask
        n_classes = pred.shape[1]
        begin = 0
        for i in range(begin, n_classes - 1):
            p = pred[:, i]

            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p) + 1e-5)
                    loss_precision = F.binary_cross_entropy_with_logits(
                        inverse_sigmoid(precision), torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target) + 1e-5)

                    loss_recall = F.binary_cross_entropy_with_logits(
                        inverse_sigmoid(recall), torch.ones_like(recall)
                    )
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum(
                        (1 - p) * (1 - completion_target)
                    ) / (torch.sum(1 - completion_target) + 1e-5)

                    loss_specificity = F.binary_cross_entropy_with_logits(
                        inverse_sigmoid(specificity),
                        torch.ones_like(specificity),
                    )
                    loss_class += loss_specificity
                loss += loss_class
        l = loss / count
        if torch.isnan(l):
            raise FloatingPointError(
                'Masked semantic scene-affinity loss is NaN.'
            )
        return l
