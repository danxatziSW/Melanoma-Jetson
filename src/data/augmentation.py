import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(input_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(input_size, input_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_val_transform(input_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=int(input_size * 1.1), width=int(input_size * 1.1)),
            A.CenterCrop(height=input_size, width=input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_seg_train_transform(input_size: int = 256) -> A.Compose:
    # applies to both image and mask
    return A.Compose(
        [
            A.RandomResizedCrop(size=(input_size, input_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.15),
            A.GridDistortion(p=0.1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_seg_val_transform(input_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=input_size, width=input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
