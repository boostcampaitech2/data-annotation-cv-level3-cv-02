import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ElasticTransform
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    Rotate,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    Resize,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    GridDropout,  # GridMask
    ChannelShuffle,
    CoarseDropout,  # Cutout
    ColorJitter
)
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists

# ToTensor를 사용하면 동작을 안합니다.

def BasicTransform():
    return A.Compose(
        [
            # RandomRotate90(),
            Resize(512, 512),
            # ToTensorV2(),
        ]
    )


def CustomTransform():
    return A.Compose(
        [
            CropNonEmptyMaskIfExists(height=256, width=256, p=0.5),
            Resize(512, 512),
            GridDropout(ratio=0.2, holes_number_x=5, holes_number_y=5, random_offset=True, p=0.5),
            # ToTensorV2(),
        ]
    )

def HardTransform():
    return A.Compose([
        Resize(512, 512),
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        GridDropout(ratio=0.2, holes_number_x=5, holes_number_y=5, random_offset=True, p=0.5),
        # GaussNoise(),
        Rotate(),
        RandomBrightnessContrast(),
        ToTensorV2()
    ])

def CutmixHardTransform():
    return A.Compose([
        Resize(512, 512),
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        Rotate(),
        RandomBrightnessContrast(),
        ToTensorV2()
    ])