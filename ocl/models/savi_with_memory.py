from functools import partial
from typing import Any, Dict
import copy
import torch
from torch import nn

from ocl.path_defaults import VIDEO, BOX, MASK
from ocl.tree_utils import get_tree_element, reduce_tree
from ocl.metrics import masks_to_bboxes

class SAVi_mem(nn.Module):
    def __init__(
        self,
        conditioning: nn.Module,
        feature_extractor: nn.Module,
        perceptual_grouping: nn.Module,
        decoder: nn.Module,
        transition_model: nn.Module,
        memory: nn.Module,
        freeze = False,
    ):
        super().__init__()
        self.conditioning = conditioning
        self.feature_extractor = feature_extractor
        self.perceptual_grouping = perceptual_grouping
        self.decoder = decoder
        self.transition_model = transition_model
        self.memory = memory
        if freeze:
            self.conditioning.eval()
            self.feature_extractor.eval()
            self.perceptual_grouping.eval()
            self.decoder.eval()
            self.transition_model.eval()
            # freeze params
            for param in self.conditioning.parameters():
                param.requires_grad = False
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.perceptual_grouping.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.transition_model.parameters():
                param.requires_grad = False
        self.batched_input = None

    def remove_bg_id(self, slot_masks):
        slot_masks = slot_masks > 0.7
        n, h, w = slot_masks.shape
        # remove background or none masks
        mask_sum = torch.sum(slot_masks.reshape(-1, h * w), dim=-1)
        empty_idx = (mask_sum <= 10).nonzero(as_tuple=True)[0].tolist()
        # bg_value = mask_sum[3]
        # bg_idx = (mask_sum == bg_value).nonzero(as_tuple=True)[0]
        # bg_idx = torch.cat([bg_idx, empty_idx], dim=0)
        fg_idxs = (mask_sum>20).nonzero(as_tuple=True)[0].tolist()
        bg_idx = 3
        fg_idx = 0
        for i in fg_idxs:
            if i not in empty_idx:
                if fg_idx != bg_idx:
                    fg_idx = i
                break
        empty_idx = list(empty_idx)
        if bg_idx in empty_idx:
            del empty_idx[empty_idx.index(3)]
        return empty_idx, fg_idx, bg_idx

    def remove_duplicated_slot_id(self, slot_masks):
        slot_masks = slot_masks > 0.7
        n, h, w = slot_masks.shape
        # remove background or none masks
        mask_sum = torch.sum(slot_masks.reshape(-1, h * w), dim=-1)
        empty_idx = (mask_sum <= 10).nonzero(as_tuple=True)[0]

        # remove duplicated masks
        mask = slot_masks.unsqueeze(1).to(torch.bool).reshape(n, 1, -1)
        mask_ = slot_masks.unsqueeze(0).to(torch.bool).reshape(1, n, -1)
        intersection = torch.sum(mask & mask_, dim=-1).to(torch.float64)
        union = torch.sum(mask | mask_, dim=-1).to(torch.float64)
        union1 = torch.sum(mask | mask, dim=-1).to(torch.float64)
        # union2 = torch.sum(mask_ | mask_, dim=-1).to(torch.float64)
        pairwise_iou = intersection / union
        self_iou1 = intersection / union1
        # self_iou2 = intersection / union2
        pairwise_iou[union == 0] = 1.0
        dup_idx = []
        for i in range(n):
            for j in range(i + 1, n):
                # if pairwise_iou[i, j] > 0.9 or self_iou1[i, j]>0.6 or self_iou2[i, j]>0.6:
                if pairwise_iou[i, j] > 0.9 or self_iou1[i, j] > 0.6:
                    dup_idx.append(i)

        invalid_idx = [*set(list(empty_idx) + list(dup_idx) +[3])]
        valid_idx = [3]
        for i in range(n):
            if i not in invalid_idx:
                valid_idx.append(i)
        return valid_idx

    def forward(self, inputs: Dict[str, Any], phase: str):

        output = inputs
        video = get_tree_element(inputs, VIDEO.split("."))
        # box = get_tree_element(inputs, BOX.split("."))

        batch_size = video.shape[0]
        features = self.feature_extractor(video=video)
        output["feature_extractor"] = features
        conditioning = self.conditioning(batch_size=batch_size)
        output["initial_conditioning"] = conditioning

        # Loop over time.
        perceptual_grouping_outputs = []
        decoder_outputs = []
        memory_outputs = []
        rollout_outputs = []
        object_features = []
        object_eval_masks= []
        objects = []
        attn_index = []
        tables = []
        slots_feats = []
        for (frame_id, frame_features) in enumerate(features):
            if frame_id == 0:
                query = conditioning
            else:
                query = conditioning

            perceptual_grouping_output = self.perceptual_grouping(
                extracted_features=frame_features, conditioning=query
            )
            slots = perceptual_grouping_output.objects
            # slots_ori = slots.clone()
            conditioning = self.transition_model(slots).clone()

            decoder_output = self.decoder(object_features=slots)
            cur_slot_masks = decoder_output.masks_eval
            bs, n = cur_slot_masks.shape[:2]
            for b in range(bs):
                empty_id, fg_id, bg_id = self.remove_bg_id(cur_slot_masks[b])
                if len(list(empty_id)) > 0:
                    tmp = slots[b,fg_id].unsqueeze(0).repeat(len(list(empty_id)),1)
                    slots[b, empty_id] = tmp

            decoder_output_new = self.decoder(object_features=slots)

            cur_slot_masks = decoder_output_new.masks_eval
            cur_slot_amodal_masks = decoder_output_new.masks

            if frame_id == 0:
                prev_slot_masks = cur_slot_masks.clone()
            memory_output = self.memory(observations=slots , prev_slot_masks = prev_slot_masks, amodal_masks = cur_slot_amodal_masks,
                cur_slot_masks = cur_slot_masks, conditions=slots, frame_id=frame_id, phase = phase)
            # prev_slot_masks = cur_slot_masks.clone()

            rollout_output = self.decoder(object_features= memory_output.rollout)
            prev_slot_masks = rollout_output.masks_eval
            object_feature_masks = self.decoder(object_features=memory_output.object_features)

            # remove background slots for evaluation
            mem_masks = object_feature_masks.masks_eval
            mem_masks_eval = torch.zeros(mem_masks.shape).to(mem_masks.device)
            num_buffer, w, h = mem_masks.shape[1:]
            for b in range(bs):
                masks = mem_masks[b] > 0.7
                boxes = masks_to_bboxes(masks)
                for i in range(num_buffer):
                    if not(boxes[i, 2] > 0.4 * w or boxes[i, 3] > 0.4*h):
                        mem_masks_eval[b, i] = mem_masks[b, i]


            # Store outputs.
            perceptual_grouping_outputs.append(perceptual_grouping_output)
            decoder_outputs.append(decoder_output_new) # decoder_output_new for memory training
            memory_outputs.append(memory_output.rollout)
            rollout_outputs.append(rollout_output)
            object_features.append(object_feature_masks)
            objects.append(memory_output.object_features)
            attn_index.append(memory_output.attn_index)
            slots_feats.append(slots)
            object_eval_masks.append(mem_masks_eval)
            tables.append(memory_output.table)

        # Stack all recurrent outputs.
        stacking_fn = partial(torch.stack, dim=1)
        output["perceptual_grouping"] = reduce_tree(perceptual_grouping_outputs, stacking_fn)
        output["decoder"] = reduce_tree(decoder_outputs, stacking_fn)
        output["memory"] = reduce_tree(memory_outputs, stacking_fn)
        output["rollout_decode"] = reduce_tree(rollout_outputs, stacking_fn)
        output["mem_masks"] = reduce_tree(object_features, stacking_fn)
        output['objects'] = reduce_tree(objects, stacking_fn)
        output["attn_index"] = reduce_tree(attn_index, stacking_fn)
        output["slots"] = reduce_tree(slots_feats, stacking_fn)
        output["tracks"] = reduce_tree(object_eval_masks, stacking_fn)
        output["table"] = reduce_tree(tables, stacking_fn)


        return output

