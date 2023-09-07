from functools import partial
from typing import Callable, Optional, Union
from ocl.matching import CPUHungarianMatcher
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from ocl import base, consistency, path_defaults, scheduling
from ocl.utils import RoutableMixin
from typing import Any
from scipy.optimize import linear_sum_assignment
from ocl.base import Instances
from torchvision.ops import generalized_box_iou
from ocl.utils import box_cxcywh_to_xyxy


def _constant_weight(weight: float, global_step: int):
    return weight


class ReconstructionLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_type: str,
        weight: Union[Callable, float] = 1.0,
        normalize_target: bool = False,
        input_path: Optional[str] = None,
        target_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {"input": input_path, "target": target_path, "global_step": path_defaults.GLOBAL_STEP},
        )
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "mse_sum":
            # Used for slot_attention and video slot attention.
            self.loss_fn = (
                lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="sum") / x1.shape[0]
            )
        elif loss_type == "l1":
            self.loss_name = "l1_loss"
            self.loss_fn = nn.functional.l1_loss
        elif loss_type == "cosine":
            self.loss_name = "cosine_loss"
            self.loss_fn = lambda x1, x2: -nn.functional.cosine_similarity(x1, x2, dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss {loss_type}. Valid choices are (mse, l1, cosine).")
        # If weight is callable use it to determine scheduling otherwise use constant value.
        self.weight = weight if callable(weight) else partial(_constant_weight, weight)
        self.normalize_target = normalize_target

    @RoutableMixin.route
    def forward(self, input: torch.Tensor, target: torch.Tensor, global_step: int):
        target = target.detach()
        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = self.loss_fn(input, target)
        weight = self.weight(global_step)
        return weight * loss


class LatentDupplicateSuppressionLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        weight: Union[float, scheduling.HPSchedulerT],
        eps: float = 1e-08,
        grouping_path: Optional[str] = "perceptual_grouping",
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self, {"grouping": grouping_path, "global_step": path_defaults.GLOBAL_STEP}
        )
        self.weight = weight
        self.similarity = nn.CosineSimilarity(dim=-1, eps=eps)

    @RoutableMixin.route
    def forward(self, grouping: base.PerceptualGroupingOutput, global_step: int):
        if grouping.dim() == 4:
            # Build large tensor of reconstructed video.
            # objects = grouping.objects
            objects = grouping
            bs, n_frames, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, :, off_diag_indices[0], :], objects[:, :, off_diag_indices[1], :]
                )
                ** 2
            )

            # if grouping.is_empty is not None:
            #     p_not_empty = 1.0 - grouping.is_empty
            #     # Assume that the probability of of individual objects being present is independent,
            #     # thus the probability of both being present is the product of the individual
            #     # probabilities.
            #     p_pair_present = (
            #         p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
            #     )
            #     # Use average expected penalty as loss for each frame.
            #     losses = (sq_similarities * p_pair_present) / torch.sum(
            #         p_pair_present, dim=-1, keepdim=True
            #     )
            # else:
            losses = sq_similarities.mean(dim=-1)

            weight = self.weight(global_step) if callable(self.weight) else self.weight

            return weight * losses.sum() / (bs * n_frames)
        elif grouping.dim() == 3:
            # Build large tensor of reconstructed image.
            objects = grouping
            bs, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, off_diag_indices[0], :], objects[:, off_diag_indices[1], :]
                )
                ** 2
            )

            if grouping.is_empty is not None:
                p_not_empty = 1.0 - grouping.is_empty
                # Assume that the probability of of individual objects being present is independent,
                # thus the probability of both being present is the product of the individual
                # probabilities.
                p_pair_present = (
                    p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
                )
                # Use average expected penalty as loss for each frame.
                losses = (sq_similarities * p_pair_present) / torch.sum(
                    p_pair_present, dim=-1, keepdim=True
                )
            else:
                losses = sq_similarities.mean(dim=-1)

            weight = self.weight(global_step) if callable(self.weight) else self.weight
            return weight * losses.sum() / bs
        else:
            raise ValueError("Incompatible input format.")


class EM_loss(nn.Module, RoutableMixin):
    def __init__(
            self,
            loss_weight: float = 20,
            pred_mask_path: Optional[str] = None,
            rec_path: Optional[str] = None,
            tgt_mask_path: Optional[str] = None,
            img_path: Optional[str] = None,
            tgt_vis_path: Optional[str] = None,
            attn_index_path: Optional[str] = None,
            pred_feat_path: Optional[str] = None,
            gt_feat_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "pred_mask": pred_mask_path,
                "reconstructions": rec_path,
                "tgt_mask": tgt_mask_path,
                "masks_vis": tgt_vis_path,
                "rec_tgt": img_path,
                "pred_feats": pred_feat_path,
                "gt_feats": gt_feat_path,
                "attn_index": attn_index_path,
            },
        )
        self.loss_weight = loss_weight
        self.loss_fn = (
            lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="none")
        )

    @RoutableMixin.route
    def forward(
            self,
            pred_mask: torch.Tensor,  # rollout_decode.masks
            tgt_mask: torch.Tensor,  # decoder.masks
            reconstructions: torch.Tensor,
            rec_tgt: torch.Tensor,
            masks_vis: torch.Tensor,
            pred_feats: torch.Tensor,
            gt_feats: torch.Tensor,
            attn_index: torch.Tensor,

    ):
        b, f, c, h, w = pred_mask.shape
        _, _, n_slots, n_buffer = attn_index.shape
        dim = pred_feats.shape[-1]

        pred_feats = F.normalize(pred_feats, dim=-1)
        gt_feats = F.normalize(gt_feats, dim=-1)

        pred_feats = pred_feats.reshape(-1, n_buffer, dim).unsqueeze(1).repeat(1, n_slots, 1, 1)
        gt_feats = gt_feats.reshape(-1, n_slots, dim).unsqueeze(2).repeat(1, 1, n_buffer, 1)

        pred_mask = pred_mask.reshape(-1, n_buffer, h, w).unsqueeze(1).repeat(1,n_slots,1,1,1)
        tgt_mask = tgt_mask.reshape(-1, n_slots, h, w).unsqueeze(2).repeat(1,1,n_buffer,1,1)
        tgt_mask = tgt_mask > 0.5
        masks_vis = masks_vis.reshape(-1, n_slots, h, w).unsqueeze(2).unsqueeze(3).repeat(1,1,n_buffer,3,1,1)
        masks_vis = masks_vis > 0.5
        attn_index = attn_index.reshape(-1, n_slots, n_buffer)
        rec_tgt = rec_tgt.reshape(-1,3,h,w).unsqueeze(1).unsqueeze(2).repeat(1,n_slots,n_buffer,1,1,1)
        reconstructions = reconstructions.reshape(-1, n_buffer, 3, h, w).unsqueeze(1).repeat(1,n_slots,1,1,1,1)
        rec_pred = reconstructions * masks_vis
        rec_tgt_ = rec_tgt * masks_vis
        loss = torch.sum(F.binary_cross_entropy(pred_mask, tgt_mask.float(), reduction = 'none'), (-1,-2)) / (h*w) + 0.1 * torch.sum(self.loss_fn(rec_pred, rec_tgt_), (-3,-2,-1))
        #loss = torch.sum(self.loss_fn(pred_feats, gt_feats), -1)
        total_loss = torch.sum(attn_index * loss, (0,1,2)) / (b * f * n_slots * n_buffer)
        return (total_loss) * self.loss_weight





