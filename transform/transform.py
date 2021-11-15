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
    Blur,
    GaussianBlur,
    GaussNoise,
    GridDropout,  # GridMask
    ChannelShuffle,
    CoarseDropout,  # Cutout
    ColorJitter,
    Normalize,
    InvertImg,
    Sharpen
)
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists

# ToTensor를 사용하면 동작을 안합니다.

def NoTransform():
    return None

def DefaultTransform():
    return A.Compose(
        [
            ColorJitter(0.5, 0.5, 0.5, 0.25),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

# variable comtinations of transform for sweep
def T1():
    return A.Compose(
        [
            Rotate(limit=80, p=0.5),
            ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.3),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )
def T1_origin():
    return A.Compose(
        [
            Rotate(limit=80, p=0.5),
            ColorJitter(0.5, 0.5, 0.5, 0.25),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

def T2():
    return A.Compose(
        [
            Sharpen(p=0.5), # sharpening
            ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.3),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )

def T2_origin():
    return A.Compose(
        [
            Sharpen(p=0.5), # sharpening
            ColorJitter(0.5, 0.5, 0.5, 0.25),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

def T3():
    return A.Compose(
        [
            Blur(p=0.5), 
            ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.3),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )

def T3_origin():
    return A.Compose(
        [
            Blur(p=0.5), 
            ColorJitter(0.5, 0.5, 0.5, 0.25),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

def T4():
    return A.Compose(
        [
            InvertImg(p=0.5), 
            ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.3),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )

def T4():
    return A.Compose(
        [
            InvertImg(p=0.5), 
            ColorJitter(0.5, 0.5, 0.5, 0.25),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

def BasicTransform():
    return A.Compose(
        [
            ColorJitter(0.5, 0.5, 0.5, 0.25),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            RandomBrightnessContrast(),
            # RandomRotate90(),
            # Resize(512, 512),
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
        # Resize(512, 512),
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        GridDropout(ratio=0.2, holes_number_x=5, holes_number_y=5, random_offset=True, p=0.5),
        # GaussNoise(),
        Rotate(),
        RandomBrightnessContrast(),
        # ToTensorV2()
    ])

def CutmixHardTransform():
    return A.Compose([
        # Resize(512, 512),
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        Rotate(),
        RandomBrightnessContrast(),
        # ToTensorV2()
    ])