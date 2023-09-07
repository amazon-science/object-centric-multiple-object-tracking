from functools import partial
from typing import Any, Dict
import copy
import torch
from torch import nn

from ocl.path_defaults import VIDEO, BOX
from ocl.tree_utils import get_tree_element, reduce_tree


class SAVi(nn.Module):
    def __init__(
        self,
        conditioning: nn.Module,
        feature_extractor: nn.Module,
        perceptual_grouping: nn.Module,
        decoder: nn.Module,
        transition_model: nn.Module,
    ):
        super().__init__()
        self.conditioning = conditioning
        self.feature_extractor = feature_extractor
        self.perceptual_grouping = perceptual_grouping
        self.decoder = decoder
        self.transition_model = transition_model
        self.batched_input = None

    def forward(self, inputs: Dict[str, Any], phase = 'train'):
        # if self.batched_input is None:
        #     video = get_tree_element(inputs, VIDEO.split("."))
        #     # if video.shape[1] == 6:
        #     self.batched_input = copy.deepcopy(inputs)
        # else:
        #     print ('use catched')
        #     inputs = self.batched_input

        output = inputs
        video = get_tree_element(inputs, VIDEO.split("."))
        box = get_tree_element(inputs, BOX.split("."))
        batch_size = video.shape[0]

        features = self.feature_extractor(video=video)
        output["feature_extractor"] = features
        conditioning = self.conditioning(batch_size=batch_size)
        # conditioning = self.conditioning(batch_size=batch_size)
        output["initial_conditioning"] = conditioning

        # Loop over time.
        perceptual_grouping_outputs = []
        decoder_outputs = []
        transition_model_outputs = []
        trackers = []
        for frame_features in features:
            perceptual_grouping_output = self.perceptual_grouping(
                extracted_features=frame_features, conditioning=conditioning
            )
            slots = perceptual_grouping_output.objects
            decoder_output = self.decoder(object_features=slots)

            # remove background
            masks = decoder_output.masks_eval
            valid_idx = [0,1,2,4,5,6,7,8,9,10]
            masks_obj = masks[:, valid_idx]

            conditioning = self.transition_model(slots)
            # Store outputs.
            perceptual_grouping_outputs.append(slots)
            decoder_outputs.append(decoder_output)
            transition_model_outputs.append(conditioning)
            trackers.append(masks_obj)

        # Stack all recurrent outputs.
        stacking_fn = partial(torch.stack, dim=1)
        output["perceptual_grouping"] = reduce_tree(perceptual_grouping_outputs, stacking_fn)
        output["decoder"] = reduce_tree(decoder_outputs, stacking_fn)
        output["transition_model"] = reduce_tree(transition_model_outputs, stacking_fn)
        output["tracks"] = reduce_tree(trackers, stacking_fn)
        return output

