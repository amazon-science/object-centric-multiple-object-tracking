import torch
import torch.nn as nn
import torch.nn.functional as F
from ocl.utils import RoutableMixin
from ocl.predictor import MLP
import dataclasses
from typing import Optional
from ocl.memory_rollout import GPT
from ocl import base, path_defaults
from ocl.mha import MultiHeadAttention, MultiHeadAttention_for_index


@dataclasses.dataclass
class MemoryOutput:
    rollout: torch.Tensor  # noqa: F821
    object_features: torch.Tensor  # noqa: F821
    mem: torch.Tensor
    eval_mem_features: torch.Tensor
    table: torch.Tensor
    attn_index: torch.Tensor


class SelfSupervisedMemory(nn.Module, RoutableMixin):
    def __init__(self,
                 embed_dim: int = 128,
                 num_objects: int = 20,
                 memory_len: int = 30,
                 stale_number: int = 5,
                 object_features_path: Optional[str] = path_defaults.OBJECTS,
                 conditioning_path: Optional[str] = path_defaults.CONDITIONING,
                 attention_maps_path: Optional[str] = None,
                 frame_features_path: Optional[str] = path_defaults.FEATURES,
                 first_box_path: Optional[str] = None,
                 init_flag_path: Optional[str] = None,
                 matched_idx_path: Optional[str] = None,
                 # eval_flag: Optional[str] = False,
                 ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(self, {
            "observation": object_features_path,
            "conditions": conditioning_path,
            "attention_maps": attention_maps_path,
            "frame_features": frame_features_path,
            "first_frame_boxes": first_box_path,
            "init_flag": init_flag_path,
            "matched_index": matched_idx_path,
        }
                               )
        self.embed_dim = embed_dim
        self.memory_len = memory_len
        self.object_num = num_objects
        self.stale_number = stale_number
        self.num_heads = 4
        self.roll_out_module = GPT(buffer_len=memory_len, n_layer=8, n_head=8, n_embd = embed_dim)
        self.register_buffer("memory", torch.zeros(8, 7, 12, embed_dim))
        self.register_buffer("memory_table", torch.zeros(8, 12))
        self.register_buffer("stale_counter", torch.zeros(8, 12))
        self.MultiHead_1 = MultiHeadAttention_for_index(n_head = 4, d_model = embed_dim, d_k = embed_dim, d_v = embed_dim)
        self.MultiHead_2 = MultiHeadAttention(n_head = 4, d_model = embed_dim, d_k = embed_dim, d_v = embed_dim)

    def remove_duplicated_slot_id_by_box_intersection(self, slot_masks):
        slot_masks = slot_masks>0.5
        n, h, w = slot_masks.shape
        mask_sum = torch.sum(slot_masks.reshape(-1, h * w), dim=-1)
        empty_idx = (mask_sum <= 10).nonzero(as_tuple=True)[0]
        bg_idx = [3]

        dup_idx = []
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
        self_iou1[union == 0] = 1.0
        # self_iou2[union == 0] = 1.0

        for i in range(n):
            for j in range(i + 1, n):
                # if pairwise_iou[i, j] > 0.9 or self_iou1[i, j]>0.6 or self_iou2[i, j]>0.6:
                if pairwise_iou[i, j] > 0.9 or self_iou1[i, j] > 0.6:
                    dup_idx.append(i)

        valid_idx = []
        invalid_idx = [*set(list(empty_idx) + list(dup_idx)+ bg_idx)]
        for i in range(n):
            if i not in invalid_idx:
                valid_idx.append(i)
        return valid_idx


    def remove_duplicated_slot_id(self, slot_masks, iou_threshold = 0.5):
        slot_masks = slot_masks>0.5
        n, h, w = slot_masks.shape
        # remove background or none masks
        mask_sum = torch.sum(slot_masks.reshape(-1, h * w), dim=-1)
        bg_value = mask_sum[0]
        # bg_idx = (mask_sum == bg_value).nonzero(as_tuple=True)[0]
        bg_idx = (mask_sum > 0.15 * h * w).nonzero(as_tuple=True)[0]
        empty_idx = (mask_sum <= 10).nonzero(as_tuple=True)[0]

        # remove duplicated masks
        mask = slot_masks.unsqueeze(1).to(torch.bool).reshape(n, 1, -1)
        mask_ = slot_masks.unsqueeze(0).to(torch.bool).reshape(1, n, -1)
        intersection = torch.sum(mask & mask_, dim=-1).to(torch.float64)
        union = torch.sum(mask | mask_, dim=-1).to(torch.float64)
        pairwise_iou = intersection / union
        pairwise_iou[union == 0] = 1.0
        dup_idx = []
        for i in range(n):
            for j in range(i + 1, n):
                if pairwise_iou[i, j] > iou_threshold:
                    dup_idx.append(i)
        invalid_idx = [*set(list(bg_idx) + list(empty_idx)+ list(dup_idx))]
        valid_idx = [0]
        for i in range(n):
            if i not in invalid_idx:
                valid_idx.append(i)
        # if len(empty_idx) == n:
        #     valid_idx.append(0)

        return valid_idx



    def initialization(self, conditions, cur_slots, cur_slot_masks, prev_slot_masks, amodal_masks, frame_id):
        if frame_id == 0:
            # For each video, we should initialize the register buffers as zero
            bs = conditions.shape[0]
            memory_shape = (bs, self.memory_len, self.object_num, self.embed_dim)
            memory_table_shape = (bs, self.object_num)
            stale_counter_shape = (bs, self.object_num)
            self.memory = torch.zeros(memory_shape).to(conditions.device)
            self.memory_table = torch.zeros(memory_table_shape).to(conditions.device)
            self.stale_counter = torch.zeros(stale_counter_shape).to(conditions.device)
            for b in range(bs):
                valid_idx = self.remove_duplicated_slot_id_by_box_intersection(amodal_masks[b])
                num_obj = len(valid_idx)
                self.memory[b, 0, 0, :] = conditions[b, 3, :]
                self.memory[b, 0, 1:num_obj+1, :] = conditions[b, valid_idx, :]
                self.memory_table[b, :num_obj+1] += 1

        else:
            '''IoU score to find new objects'''
            bs, n, h, w = prev_slot_masks.shape
            for b in range(bs):
                # self.memory_eval[b, frame_id, -1, :] = ori_slots[b, 0, :]
                cur_valid_idx = self.remove_duplicated_slot_id(cur_slot_masks[b])
                pre_valid_idx = self.remove_duplicated_slot_id(prev_slot_masks[b])

                cur_slot_mask = cur_slot_masks[b][cur_valid_idx] > 0.7
                prev_slot_mask = prev_slot_masks[b][pre_valid_idx] > 0.7

                # calculate pairwise iou
                cur_mask = cur_slot_mask.unsqueeze(1).to(torch.bool).reshape(len(cur_valid_idx), 1, -1)
                prev_mask = prev_slot_mask.unsqueeze(0).to(torch.bool).reshape(1, len(pre_valid_idx), -1)
                intersection = torch.sum(cur_mask & prev_mask, dim=-1).to(torch.float64)
                union = torch.sum(cur_mask | prev_mask, dim=-1).to(torch.float64)
                pairwise_iou = intersection / union
                pairwise_iou[union == 0] = 1.0
                sim, _ = torch.max(pairwise_iou, dim=-1)
                # new_obj_idx = list((sim < 10).nonzero(as_tuple=True)[0])
                new_obj_idx = (sim ==0).nonzero(as_tuple=True)[0].tolist()

                new_obj_idx_ori = [cur_valid_idx[id] for id in new_obj_idx]
                num_new_obj = len(new_obj_idx_ori)

                new_mem_idx = (self.memory_table[b] == 0).nonzero(as_tuple=True)[0].tolist()
                old_mem_idx = (self.memory_table[b] != 0).nonzero(as_tuple=True)[0].tolist()
                if num_new_obj >0 and len(new_mem_idx) > 0:
                    last_pos = old_mem_idx[-1]+1
                    if last_pos + num_new_obj - 1 in new_mem_idx:
                        self.memory[b, 0, last_pos: last_pos + num_new_obj] = cur_slots[b, new_obj_idx_ori]
                        # self.memory_eval[b, 0, last_pos: last_pos + num_new_obj] = cur_slots[b, new_obj_idx_ori]
                        self.memory_table[b, last_pos: last_pos + num_new_obj] += 1


    def buffer_terminate(self):
        bs = self.stale_counter.shape[0]
        for b in range(bs):
            terminate_idx = (self.stale_counter[b] >= self.stale_number).nonzero(as_tuple=True)[0]
            if len(terminate_idx)>0:
                num_dead_buffer = len(list(terminate_idx))
                tmp = torch.zeros((self.memory_len, num_dead_buffer, self.embed_dim)).to(self.memory.device)
                self.memory[b, :, terminate_idx] = tmp
                self.stale_counter[b, terminate_idx] = 0



    def sms_attn(self, observations, predictions, attn_mask, eval_flag):
        attn_o_to_p, attn_o_to_p_weights = self.MultiHead_1(observations, predictions, predictions, mask = attn_mask)


        mask = torch.zeros(attn_o_to_p_weights.shape).to(attn_o_to_p_weights.device)
        b, w, h = mask.shape
        for i in range(b):
            for j in range(w):
                index = torch.argmax(attn_o_to_p_weights[i, j, :])
                mask[i, j, index] = 1



        weights = mask.clone()

        # MultiHead_2 layer
        if not eval_flag:
            attn_o_to_p_weights_trans = torch.transpose(attn_o_to_p_weights, 1, 2)
        else:
            attn_o_to_p_weights_trans = torch.transpose(weights, 1, 2)

        attn_p_to_o, attn_p_to_o_weights = self.MultiHead_2(predictions, observations, observations, mask=attn_o_to_p_weights_trans)

        # replace the attn_p_to_o with predictions if the buffer is not assigned
        if eval_flag:
            b, h, w = weights.shape
            weights_new = torch.zeros((b, h + 1, w)).to(attn_o_to_p.device)  # [b, n+1, m]
            weights_new[:, 0:h, :] = weights
            weights_new[:, h, :] = torch.sum(weights, dim=1)
            weights_new_convert_zero = weights_new[:, h, :].clone()
            weights_new_convert_zero[weights_new[:, h, :] == 0] = 1
            weights_new_convert_zero[weights_new[:, h, :] > 0] = 0
            weights_new[:, h, :] = weights_new_convert_zero
            b_p, h_p, w_p = attn_p_to_o.shape # merged features
            for j in range(b_p):
                index = weights_new[j, h, :].nonzero(as_tuple=True)[0]  # hard index
                if len(index) > 0:
                    # update the buffer that no slots matched with zero embeddings
                    attn_p_to_o[j][index] = torch.zeros((len(index), self.embed_dim)).to(observations.device)
                    # attn_p_to_o[j][index] = predictions[j][index].clone()
        else:
            b_p, h_p, w_p = attn_p_to_o.shape # merged features
            for j in range(b_p):
                # index = weights_new[j, h, :].nonzero(as_tuple=True)[0]  # hard index
                index = (self.memory_table[j] == 0).nonzero(as_tuple=True)[0]
                if len(index) > 0:
                    # update the buffer that no slots matched with zero embeddings
                    attn_p_to_o[j][index] = torch.zeros((len(index), self.embed_dim)).to(observations.device)
        return attn_p_to_o, weights, attn_o_to_p_weights, attn_p_to_o_weights




    def update_sms(self, object_features):
        object_features_ = object_features.clone().detach()

        for b in range(object_features_.shape[0]):
            for i in range(object_features_.shape[1]):
                tmp = torch.sum(object_features_[b, i, :], dim=-1)
                if tmp != 0 and self.memory_table[b, i] != 0:
                    pos = self.memory_table[b, i].cpu().numpy().astype(int)
                    if pos == self.memory_len-2:
                        self.memory[b, :-2, i] = self.memory[b, 1:-1, i]
                        self.memory[b, -2, i] = object_features_[b, i]
                    else:
                        self.memory[b, pos, i] = object_features_[b, i]
                        self.memory_table[b, i] += 1
                else:
                    self.stale_counter[b, i] += 1

        return object_features

    def generate_mem_attnmask(self, memory_table, slots):
        bs, num_buffer = memory_table.shape
        num_slot = slots.shape[1]
        attn_mask = torch.zeros((bs, num_slot, num_buffer)).to(memory_table.device)
        for b in range(bs):
            tmp = torch.sum(slots[b], dim=-1)
            valid_slot_idx = (tmp!=0).nonzero(as_tuple=True)[0].tolist()
            valid_buffer_idx = (self.memory_table[b] != 0).nonzero(as_tuple=True)[0].tolist()
            if len(valid_slot_idx) > 0 and len(valid_buffer_idx) > 0 :
                num_slot_valid = len(valid_slot_idx)
                num_buffer_valid = len(valid_buffer_idx)
                attn_mask[b, :num_slot_valid, :num_buffer_valid] += 1
        return attn_mask


    @RoutableMixin.route
    def forward(self,
                observations: torch.Tensor,
                prev_slot_masks: torch.Tensor,
                cur_slot_masks: torch.Tensor,
                amodal_masks: torch.Tensor,
                conditions: torch.Tensor,
                frame_id: int,
                phase: str,
                ):
        if phase == 'train':
            eval = False
        elif phase == 'val':
            eval = True
        self.initialization(conditions, observations, cur_slot_masks, prev_slot_masks, amodal_masks, frame_id)
        if frame_id == 0:
            predictions = self.memory[:, 0].clone()
            object_features = predictions.clone()
            b, n_slots = observations.shape[:2]
            n_buffer = predictions.shape[1]
            attn_index = torch.zeros((b, n_slots, n_buffer)).to(observations.device)

        else:
            predictions = self.roll_out_module(self.memory, self.memory_table)

            predictions[:, 0] = observations[:, 3] # index 3 denotes the background slot

            attn_mask = self.generate_mem_attnmask(self.memory_table, observations)
            object_features, weights, attn_index, mask_o_p = self.sms_attn(observations, predictions, attn_mask, eval_flag=eval)
            object_features[:, 0] = observations[:, 3]

            # memory update
            _ = self.update_sms(object_features)

        # memory terminate
        # if eval:
        #     self.buffer_terminate()
        return MemoryOutput(
            rollout=predictions,
            object_features=object_features,
            mem = self.memory,
            eval_mem_features = object_features,
            table = self.memory_table,
            attn_index=attn_index,
        )


