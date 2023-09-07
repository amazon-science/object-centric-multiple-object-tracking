from typing import Callable, Dict, List, Optional, Tuple

import torch
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks, make_grid, draw_bounding_boxes
import numpy as np
from ocl import consistency, visualization_types
from ocl.utils import RoutableMixin, box_cxcywh_to_xyxy
from ocl.metrics import masks_to_bboxes_xyxy
from torchvision.ops import masks_to_boxes

def _nop(arg):
    return arg


class Image(RoutableMixin):
    def __init__(
        self,
        n_instances: int = 8,
        n_row: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        as_grid: bool = True,
        image_path: Optional[str] = None,
    ):
        super().__init__({"image": image_path})
        self.n_instances = n_instances
        self.n_row = n_row
        self.denormalization = denormalization if denormalization else _nop
        self.as_grid = as_grid

    @RoutableMixin.route
    def __call__(self, image: torch.Tensor):
        image = self.denormalization(image[: self.n_instances].cpu())
        if self.as_grid:
            return visualization_types.Image(make_grid(image, nrow=self.n_row))
        else:
            return visualization_types.Images(image)


class Video(RoutableMixin):
    def __init__(
        self,
        n_instances: int = 8,
        n_row: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        as_grid: bool = True,
        video_path: Optional[str] = None,
    ):
        super().__init__({"video": video_path})
        self.n_instances = n_instances
        self.n_row = n_row
        self.denormalization = denormalization if denormalization else _nop
        self.as_grid = as_grid

    @RoutableMixin.route
    def __call__(self, video: torch.Tensor):
        video = video[: self.n_instances].cpu()
        if self.as_grid:
            video = torch.stack(
                [
                    make_grid(self.denormalization(frame.unsqueeze(1)).squeeze(1), nrow=self.n_row)
                    for frame in torch.unbind(video, 1)
                ],
                dim=0,
            ).unsqueeze(0)
        return visualization_types.Video(video)


class Mask(RoutableMixin):
    def __init__(
        self,
        n_instances: int = 8,
        mask_path: Optional[str] = None,
    ):
        super().__init__({"masks": mask_path})
        self.n_instances = n_instances

    @RoutableMixin.route
    def __call__(self, masks):
        masks = masks[: self.n_instances].cpu().contiguous()
        image_shape = masks.shape[-2:]
        n_objects = masks.shape[-3]

        if masks.dim() == 5:
            # Handling video data.
            # bs x frames x objects x H x W
            mask_vis = masks.transpose(1, 2).contiguous()
            flattened_masks = mask_vis.flatten(0, 1).unsqueeze(2)

            # Draw masks inverted as they are easier to print.
            mask_vis = torch.stack(
                [
                    make_grid(1.0 - masks, nrow=n_objects)
                    for masks in torch.unbind(flattened_masks, 1)
                ],
                dim=0,
            )
            mask_vis = mask_vis.unsqueeze(0)
            return visualization_types.Video(mask_vis)
        elif masks.dim() == 4:
            # Handling image data.
            # bs x objects x H x W
            # Monochrome image with single channel.
            masks = masks.view(-1, 1, *image_shape)
            # Draw masks inverted as they are easier to print.
            return visualization_types.Image(make_grid(1.0 - masks, nrow=n_objects))


class VisualObject(RoutableMixin):
    def __init__(
        self,
        n_instances: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        object_path: Optional[str] = None,
        mask_path: Optional[str] = None,
    ):
        super().__init__({"object_reconstructions": object_path, "object_masks": mask_path})
        self.n_instances = n_instances
        self.denormalization = denormalization if denormalization else _nop

    @RoutableMixin.route
    def __call__(self, object_reconstructions, object_masks):
        objects = object_reconstructions[: self.n_instances].cpu()
        masks = object_masks[: self.n_instances].cpu().contiguous()
        masks = masks > 0.5

        image_shape = objects.shape[-3:]
        n_objects = objects.shape[-4]

        if objects.dim() == 6:
            # Handling video data.
            # bs x frames x objects x C x H x W

            # We need to denormalize prior to constructing the grid, yet the denormalization
            # method assumes video input. We thus convert a frame into a single frame video and
            # remove the additional dimension prior to make_grid.
            # Switch object and frame dimension.
            object_vis = objects.transpose(1, 2).contiguous()
            mask_vis = masks.transpose(1, 2).contiguous()
            flattened_masks = mask_vis.flatten(0, 1).unsqueeze(2).float()
            object_vis = self.denormalization(object_vis.flatten(0, 1))
            # Keep object pixels and apply white background to non-objects parts.
            object_vis = object_vis * flattened_masks + (1.0 - flattened_masks)

            # object_vis = object_vis * flattened_masks
            object_vis = torch.stack(
                [
                    make_grid(
                        object_vis_frame,
                        nrow=n_objects,
                    )
                    for object_vis_frame in torch.unbind(object_vis, 1)
                ],
                dim=0,
            )
            # Add batch dimension as this is required for video input.
            object_vis = object_vis.unsqueeze(0)

            # Draw masks inverted as they are easier to print.
            flattened_masks_ = flattened_masks > 0.5
            mask_vis = torch.stack(
                [
                    # make_grid(1.0 - masks, nrow=n_objects)
                    make_grid(masks, nrow=n_objects)
                    for masks in torch.unbind(flattened_masks_, 1)
                ],
                dim=0,
            )
            mask_vis = mask_vis.unsqueeze(0)
            return {
                "reconstruction": visualization_types.Video(object_vis),
                "mask": visualization_types.Video(mask_vis),
            }
        elif objects.dim() == 5:
            # Handling image data.
            # bs x objects x C x H x W
            object_reconstructions = self.denormalization(objects.view(-1, *image_shape))
            # Monochrome image with single channel.
            masks = masks.view(-1, 1, *image_shape[1:])
            # Save object reconstructions as RGBA image. make_grid does not support RGBA input, thus
            # we combine the channels later.  For the masks we need to pad with 1 as we want the
            # borders between images to remain visible (i.e. alpha value of 1.)
            masks_grid = make_grid(masks, nrow=n_objects, pad_value=1.0)
            object_grid = make_grid(object_reconstructions, nrow=n_objects)
            # masks_grid expands the image to three channels, which we don't need. Only keep one, and
            # use it as the alpha channel. After make_grid the tensor has the shape C X W x H.
            object_grid = torch.cat((object_grid, masks_grid[:1]), dim=0)

            return {
                "reconstruction": visualization_types.Image(object_grid),
                # Draw masks inverted as they are easier to print.
                "mask": visualization_types.Image(make_grid(1.0 - masks, nrow=n_objects)),
            }


class ConsistencyMask(RoutableMixin):
    def __init__(
        self,
        matcher: consistency.HungarianMatcher,
        mask_path: Optional[str] = None,
        mask_target_path: Optional[str] = None,
        params_path: Optional[str] = None,
    ):
        super().__init__(
            {"mask": mask_path, "mask_target": mask_target_path, "cropping_params": params_path}
        )
        self.matcher = matcher

    @RoutableMixin.route
    def __call__(self, mask: torch.Tensor, mask_target: torch.Tensor, cropping_params: torch.Tensor):
        _, _, size, _ = mask.shape
        mask_one_hot = self._to_binary_mask(mask)
        mask_target = self.crop_views(mask_target, cropping_params, size)
        mask_target_one_hot = self._to_binary_mask(mask_target)
        _ = self.matcher(mask_one_hot, mask_target_one_hot)
        return {
            "costs": visualization_types.Image(
                make_grid(-self.matcher.costs, nrow=8, pad_value=0.9)
            ),
        }

    @staticmethod
    def _to_binary_mask(masks: torch.Tensor):
        _, n_objects, _, _ = masks.shape
        m_lables = masks.argmax(dim=1)
        mask_one_hot = torch.nn.functional.one_hot(m_lables, n_objects)
        return mask_one_hot.permute(0, 3, 1, 2)

    def crop_views(self, view: torch.Tensor, param: torch.Tensor, size: int):
        return torch.cat([self.crop_maping(v, p, size) for v, p in zip(view, param)])

    @staticmethod
    def crop_maping(view: torch.Tensor, p: torch.Tensor, size: int):
        p = tuple(p.cpu().numpy().astype(int))
        return transforms.functional.resized_crop(view, *p, size=(size, size))[None]


class Segmentation(RoutableMixin):
    def __init__(
        self,
        n_instances: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_path: Optional[str] = None,
        mask_path: Optional[str] = None,
    ):
        super().__init__({"image": image_path, "mask": mask_path})
        self.n_instances = n_instances
        self.denormalization = denormalization if denormalization else _nop
        self._cmap_cache: Dict[int, List[Tuple[int, int, int]]] = {}

    def _get_cmap(self, num_classes: int) -> List[Tuple[int, int, int]]:
        if num_classes in self._cmap_cache:
            return self._cmap_cache[num_classes]

        from matplotlib import cm

        if num_classes <= 20:
            mpl_cmap = cm.get_cmap("tab20", num_classes)(range(num_classes))
        else:
            mpl_cmap = cm.get_cmap("turbo", num_classes)(range(num_classes))

        cmap = [tuple((255 * cl[:3]).astype(int)) for cl in mpl_cmap]
        self._cmap_cache[num_classes] = cmap
        return cmap

    @RoutableMixin.route
    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Optional[visualization_types.Visualization]:
        image = image[: self.n_instances].cpu()
        mask = mask[: self.n_instances].cpu().contiguous()
        if image.dim() == 4:  # Only support image data at the moment.
            input_image = self.denormalization(image)
            n_objects = mask.shape[1]

            masks_argmax = mask.argmax(dim=1)[:, None]
            classes = torch.arange(n_objects)[None, :, None, None].to(masks_argmax)
            masks_one_hot = masks_argmax == classes

            cmap = self._get_cmap(n_objects)
            masks_on_image = torch.stack(
                [
                    draw_segmentation_masks(
                        (255 * img).to(torch.uint8), mask, alpha=0.75, colors=cmap
                    )
                    for img, mask in zip(input_image.to("cpu"), masks_one_hot.to("cpu"))
                ]
            )

            return visualization_types.Image(make_grid(masks_on_image, nrow=8))
        return None

class ObjectMOT(RoutableMixin):
    def __init__(
        self,
        n_clips: int = 3,
        n_row: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        video_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        pred_is_box: bool = False,
    ):
        super().__init__({"video": video_path, "object_masks": mask_path})
        self.n_clips = n_clips
        self.n_row = n_row
        self.denormalization = denormalization if denormalization else _nop
        self.pred_is_box = pred_is_box

    def generate_color_list(self, track_num):
        import random
        color_list = []
        for i in range(track_num):
            hexadecimal = ["#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
            color_list.append(hexadecimal[0])
        return color_list


    @RoutableMixin.route
    def __call__(
        self,
        video: torch.Tensor,
        object_masks: torch.Tensor,
    ) -> Optional[visualization_types.Visualization]:
        video = video[: self.n_clips].cpu()
        num_frames = video.shape[1]

        if not self.pred_is_box:
            masks = object_masks[: self.n_clips].cpu().contiguous()
            B, F, C, h, w = masks.shape  # [5, 6, 11, 64, 64]
            masks = masks.flatten(0, 1)
            masks = masks > 0.7
            bbox = masks_to_bboxes_xyxy(masks.flatten(0, 1)).unflatten(0, (B, F, C))
        else:
            bbox = object_masks[: self.n_clips].cpu().contiguous()
            bbox[:,:,:,2] += bbox[:,:,:,0]
            bbox[:, :, :, 3] += bbox[:, :, :, 1]

        rendered_video = torch.zeros_like(video)

        n_colors = 500
        color_list = self.generate_color_list(n_colors)

        for cidx in range(self.n_clips):
            for fidx in range(num_frames):
                cur_obj_box = bbox[cidx, fidx][:, 0] != -1.0
                cur_obj_idx = cur_obj_box.nonzero()[:, 0].detach().cpu().numpy()
                idx = cur_obj_idx.tolist()

                cur_obj_idx = np.array(idx)
                cur_color_list = [color_list[obj_idx] for obj_idx in idx]
                frame = (video[cidx, fidx] * 256).to(torch.uint8)
                frame = draw_bounding_boxes(
                    frame, bbox[cidx, fidx][cur_obj_idx], colors=cur_color_list
                )
                rendered_video[cidx, fidx] = frame

        rendered_video = (
            torch.stack(
                [
                    make_grid(self.denormalization(frame.unsqueeze(1)).squeeze(1), nrow=self.n_row)
                    for frame in torch.unbind(rendered_video, 1)
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(torch.float32)
            / 256
        )

        return visualization_types.Video(rendered_video)
