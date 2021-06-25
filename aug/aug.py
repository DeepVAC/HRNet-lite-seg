from deepvac.aug import Composer
from deepvac.aug.factory import AugFactory

class LiteHRNetTrainComposer(Composer):
    def __init__(self, deepvac_config):
        super(LiteHRNetTrainComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('SpeckleAug@0.1 => GaussianAug@0.1 => HorlineAug@0.1 => VerlineAug@0.1 => LRmotionAug@0.1 => UDmotionAug@0.1 \
            => NoisyAug@0.1 => DarkAug@0.1 => ColorJitterAug@0.15 => BrightnessJitterAug@0.15 => ContrastJitterAug@0.15 => \
            ImageWithMasksRandomRotateAug@0.6 => ImageWithMasksNormalizeAug => ImageWithMasksCenterCropAug => ImageWithMasksScaleAug => \
            ImageWithMasksHFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)
        self.addAugFactory('ac1', ac1)

class LiteHRNetValComposer(Composer):
    def __init__(self, deepvac_config):
        super(LiteHRNetValComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksScaleAug => ImageWithMasksToTensorAug', deepvac_config)
        self.addAugFactory('ac1', ac1)