"""Implementation of conditioning approaches for slots."""
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ocl import base, path_defaults
from ocl.utils import RoutableMixin
from ocl.predictor import MLP

class RandomConditioning(base.Conditioning, RoutableMixin):
    """Random conditioning with potentially learnt mean and stddev."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Optional[Callable[[torch.Tensor], None]] = None,
        logsigma_init: Optional[Callable[[torch.Tensor], None]] = None,
        batch_size_path: Optional[str] = path_defaults.BATCH_SIZE,
    ):
        base.Conditioning.__init__(self)
        RoutableMixin.__init__(self, {"batch_size": batch_size_path})
        self.n_slots = n_slots
        self.object_dim = object_dim

        if learn_mean:
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_mu", torch.zeros(1, 1, object_dim))

        if learn_std:
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_logsigma", torch.zeros(1, 1, object_dim))

        if mean_init is None:
            mean_init = nn.init.xavier_uniform_
        if logsigma_init is None:
            logsigma_init = nn.init.xavier_uniform_

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    @RoutableMixin.route
    def forward(self, batch_size: int) -> base.ConditioningOutput:
        mu = self.slots_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.n_slots, -1)
        return mu + sigma * torch.randn_like(mu)


class LearntConditioning(base.Conditioning, RoutableMixin):
    """Conditioning with a learnt set of slot initializations, similar to DETR."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        slot_init: Optional[Callable[[torch.Tensor], None]] = None,
        batch_size_path: Optional[str] = path_defaults.BATCH_SIZE,
    ):
        base.Conditioning.__init__(self)
        RoutableMixin.__init__(self, {"batch_size": batch_size_path})
        self.n_slots = n_slots
        self.object_dim = object_dim

        self.slots = nn.Parameter(torch.zeros(1, n_slots, object_dim))

        if slot_init is None:
            slot_init = nn.init.normal_

        with torch.no_grad():
            slot_init(self.slots)

    @RoutableMixin.route
    def forward(self, batch_size: int) -> base.ConditioningOutput:
        return self.slots.expand(batch_size, -1, -1)


class RandomConditioningWithQMCSampling(RandomConditioning):
    """Random conditioning with learnt mean and stddev using Quasi-Monte Carlo (QMC) samples."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Optional[Callable[[torch.Tensor], None]] = None,
        logsigma_init: Optional[Callable[[torch.Tensor], None]] = None,
        batch_size_path: Optional[str] = path_defaults.BATCH_SIZE,
    ):
        super().__init__(
            object_dim,
            n_slots,
            learn_mean,
            learn_std,
            mean_init,
            logsigma_init,
            batch_size_path=batch_size_path,
        )

        import scipy.stats  # Import lazily because scipy takes some time to import

        self.randn_rng = scipy.stats.qmc.MultivariateNormalQMC(mean=np.zeros(object_dim))

    def _randn(self, *args: Tuple[int]) -> torch.Tensor:
        n_elements = np.prod(args)
        # QMC sampler needs to sample powers of 2 numbers at a time
        n_elements_rounded2 = 2 ** int(np.ceil(np.log2(n_elements)))
        z = self.randn_rng.random(n_elements_rounded2)[:n_elements]

        return torch.from_numpy(z).view(*args, -1)

    @RoutableMixin.route
    def forward(self, batch_size: int) -> base.ConditioningOutput:
        mu = self.slots_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.n_slots, -1)

        z = self._randn(batch_size, self.n_slots).to(mu, non_blocking=True)
        return mu + sigma * z


class SlotwiseLearntConditioning(base.Conditioning, RoutableMixin):
    """Random conditioning with learnt mean and stddev for each slot.

    Removes permutation equivariance compared to the original slot attention conditioning.
    """

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        mean_init: Optional[Callable[[torch.Tensor], None]] = None,
        logsigma_init: Optional[Callable[[torch.Tensor], None]] = None,
        batch_size_path: Optional[str] = path_defaults.BATCH_SIZE,
    ):
        base.Conditioning.__init__(self)
        RoutableMixin.__init__(self, {"batch_size": batch_size_path})
        self.n_slots = n_slots
        self.object_dim = object_dim

        self.slots_mu = nn.Parameter(torch.zeros(1, n_slots, object_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, n_slots, object_dim))

        if mean_init is None:
            mean_init = nn.init.normal_
        if logsigma_init is None:
            logsigma_init = nn.init.xavier_uniform_

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    @RoutableMixin.route
    def forward(self, batch_size: int) -> base.ConditioningOutput:
        mu = self.slots_mu.expand(batch_size, -1, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, -1, -1)
        return mu + sigma * torch.randn_like(mu)


class CoordinateEncoderStateInit(base.Conditioning, RoutableMixin):
    """State init that encodes bounding box corrdinates as conditional input.
    Attributes:
        embedding_transform: A nn.Module that is applied on inputs (bounding boxes).
        prepend_background: Boolean flag' whether to prepend a special, zero-valued
            background bounding box to the input. Default: False.
        center_of_mass: Boolean flag; whether to convert bounding boxes to center
            of mass coordinates. Default: False.
        background_value: Default value to fill in the background.
    """
    def __init__(self,
                 object_dim: int,
                 prepend_background: bool = True,
                 center_of_mass: bool = False,
                 background_value: float = 0. ,
                 batch_size_path: Optional[str] = path_defaults.BATCH_SIZE,
                 ):
        base.Conditioning.__init__(self)
        RoutableMixin.__init__(self, {"batch_size": batch_size_path})

        self.embedding_transform = MLP(input_size=4, hidden_size=256,output_size=128, layernorm=None)
        self.prepend_background = prepend_background
        self.center_of_mass = center_of_mass
        self.background_value = background_value
        self.object_dim = object_dim

    @RoutableMixin.route
    def forward(self, target_bbox: torch.Tensor, batch_size: int) -> base.ConditioningOutput:
        del batch_size  # Unused.

        # inputs.shape = (batch_size, seq_len, bboxes, 4)
        inputs = target_bbox[:, 0]  # Only condition on first time step.
        # inputs.shape = (batch_size, bboxes, 4)
        if self.prepend_background:
            # Adds a fake background box [0, 0, 0, 0] at the beginning.
            batch_size = inputs.shape[0]

            # Encode the background as specified by the background_value.
            background = torch.full(
                (batch_size, 1, 4), self.background_value, dtype=inputs.dtype,
                device=inputs.get_device())

            inputs = torch.cat([background, inputs], dim=1)
            # inputs = torch.cat([inputs, background], dim=1)

        if self.center_of_mass:
            y_pos = (inputs[:, :, 0] + inputs[:, :, 2]) / 2
            x_pos = (inputs[:, :, 1] + inputs[:, :, 3]) / 2
            inputs = torch.stack([y_pos, x_pos], dim=-1)

        slots = self.embedding_transform(inputs)
        # duplicated_slots = torch.cat([slots, slots], dim=1)

        return slots

class CoordinateEncoderState_learned_Init(base.Conditioning, RoutableMixin):
    """State init that encodes bounding box corrdinates as conditional input.
    Attributes:
        embedding_transform: A nn.Module that is applied on inputs (bounding boxes).
        prepend_background: Boolean flag' whether to prepend a special, zero-valued
            background bounding box to the input. Default: False.
        center_of_mass: Boolean flag; whether to convert bounding boxes to center
            of mass coordinates. Default: False.
        background_value: Default value to fill in the background.
    """
    def __init__(self,
                 object_dim: int,
                 n_slots: int,
                 prepend_background: bool = True,
                 center_of_mass: bool = False,
                 background_value: float = 0. ,
                 slot_init: Optional[Callable[[torch.Tensor], None]] = None,
                 batch_size_path: Optional[str] = path_defaults.BATCH_SIZE,
                 ):
        base.Conditioning.__init__(self)
        RoutableMixin.__init__(self, {"batch_size": batch_size_path})

        self.embedding_transform = MLP(input_size=4, hidden_size=256,output_size=128, layernorm=None)
        self.prepend_background = prepend_background
        self.center_of_mass = center_of_mass
        self.background_value = background_value
        self.object_dim = object_dim
        self.slots = nn.Parameter(torch.zeros(1, n_slots, object_dim))
        if slot_init is None:
            slot_init = nn.init.normal_

        with torch.no_grad():
            slot_init(self.slots)


    @RoutableMixin.route
    def forward(self, target_bbox: torch.Tensor, batch_size: int) -> base.ConditioningOutput:
        del batch_size  # Unused.
        inputs = target_bbox[:, 0]  # Only condition on first time step.
        inputs_box = inputs
        batch_size, object_num, _ = inputs.shape
        if self.prepend_background:
            # Adds a fake background box [0, 0, 0, 0] at the beginning.
            batch_size = inputs.shape[0]

            # Encode the background as specified by the background_value.
            background = torch.full(
                (batch_size, 1, 4), self.background_value, dtype=inputs.dtype,
                device=inputs.get_device())

            inputs = torch.cat([background, inputs], dim=1)

        if self.center_of_mass:
            y_pos = (inputs[:, :, 0] + inputs[:, :, 2]) / 2
            x_pos = (inputs[:, :, 1] + inputs[:, :, 3]) / 2
            inputs = torch.stack([y_pos, x_pos], dim=-1)

        slots = self.embedding_transform(inputs)
        slots_learned = self.slots.expand(batch_size, -1, -1)

        # slots_output = torch.zeros(slots.shape).to(target_bbox.device)
        #
        # for i in range(batch_size):
        #     for j in range(object_num):
        #         if j == 0:
        #             slots_output[i][0] = slots[i][0]
        #         else:
        #             if torch.sum(inputs_box[i][j]) == 0:
        #                 slots_output[i][j] = slots_learned[i][j]
        #             else:
        #                 slots_output[i][j] = slots[i][j]

        slots_output = torch.cat([slots, slots_learned], dim = 1)
        return slots_output

class CoordinateEncoderState_learned_m_2(base.Conditioning, RoutableMixin):
    """State init that encodes bounding box corrdinates as conditional input.
    Attributes:
        embedding_transform: A nn.Module that is applied on inputs (bounding boxes).
        prepend_background: Boolean flag' whether to prepend a special, zero-valued
            background bounding box to the input. Default: False.
        center_of_mass: Boolean flag; whether to convert bounding boxes to center
            of mass coordinates. Default: False.
        background_value: Default value to fill in the background.
    """
    def __init__(self,
                 object_dim: int,
                 n_slots: int,
                 prepend_background: bool = True,
                 center_of_mass: bool = False,
                 background_value: float = 0. ,
                 slot_init: Optional[Callable[[torch.Tensor], None]] = None,
                 batch_size_path: Optional[str] = path_defaults.BATCH_SIZE,
                 ):
        base.Conditioning.__init__(self)
        RoutableMixin.__init__(self, {"batch_size": batch_size_path})

        self.embedding_transform = MLP(input_size=4, hidden_size=256,output_size=128, layernorm=None)
        self.prepend_background = prepend_background
        self.center_of_mass = center_of_mass
        self.background_value = background_value
        self.object_dim = object_dim
        self.slots = nn.Parameter(torch.zeros(1, 1, n_slots, object_dim))
        if slot_init is None:
            slot_init = nn.init.normal_

        with torch.no_grad():
            slot_init(self.slots)


    @RoutableMixin.route
    def forward(self, target_bbox: torch.Tensor, batch_size: int) -> base.ConditioningOutput:
        del batch_size  # Unused.
        inputs = target_bbox[:, 0]  # Only condition on first time step.
        inputs_box = inputs
        batch_size, object_num, _ = inputs.shape
        if self.prepend_background:
            # Adds a fake background box [0, 0, 0, 0] at the beginning.
            batch_size = inputs.shape[0]

            # Encode the background as specified by the background_value.
            background = torch.full(
                (batch_size, 1, 4), self.background_value, dtype=inputs.dtype,
                device=inputs.get_device())

            inputs = torch.cat([background, inputs], dim=1)

        if self.center_of_mass:
            y_pos = (inputs[:, :, 0] + inputs[:, :, 2]) / 2
            x_pos = (inputs[:, :, 1] + inputs[:, :, 3]) / 2
            inputs = torch.stack([y_pos, x_pos], dim=-1)

        slots = self.embedding_transform(inputs).unsqueeze(1)
        slots_learned = self.slots.expand(batch_size, -1, -1, -1)
        slots_output = torch.cat([slots, slots_learned], dim=1)
        return slots_output

