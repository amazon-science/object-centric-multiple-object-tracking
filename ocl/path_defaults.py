"""Default paths for different types of inputs.

These are only defined for convenience and can also be overwritten using the appropriate *_path
constructor variables of RoutableMixin subclasses.
"""
INPUT = "input"
VIDEO = f"{INPUT}.image"
BOX = f"{INPUT}.instance_bbox"
MASK = f"{INPUT}.mask"
ID = f"{INPUT}.instance_id"
BATCH_SIZE = f"{INPUT}.batch_size"
GLOBAL_STEP = "global_step"
FEATURES = "feature_extractor"
CONDITIONING = "conditioning"
# TODO(hornmax): Currently decoders are nested in the task and accept PerceptualGroupingOutput as
# input. In the future this will change and decoders should just be regular parts of the model.
OBJECTS = "perceptual_grouping.objects"
FEATURE_ATTRIBUTIONS = "perceptual_grouping.feature_attributions"
OBJECT_DECODER = "object_decoder"
