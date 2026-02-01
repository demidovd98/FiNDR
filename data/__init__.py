from data.data_stats import BIRD_STATS, DOG_STATS, FLOWER_STATS, PET_STATS, CAR_STATS, IMAGENET_STATS, CIFAR10_STATS, CIFAR100_STATS, ZERO_SHOT_TEMPLATES
from data.bird200 import build_bird200_discovery, build_bird200_test
from data.dog120 import build_dog120_discovery, build_dog120_test
from data.flower102 import build_flower102_discovery, build_flower102_test
from data.pet37 import build_pet37_discovery, build_pet37_test
from data.car196 import build_car196_discovery, build_car196_test

from data.bird200 import _transform as bird_transform
from data.car196 import _transform as car_transform
from data.dog120 import _transform as dog_transform
from data.flower102 import _transform as flower_transform
from data.pet37 import _transform as pet_transform

from .utils import random_augmentation

__all__ = [
    "DATA_STATS", "DATA_GROUPING", "DATA_TRANSFORM",
    "BIRD_STATS", "DOG_STATS", "FLOWER_STATS", "PET_STATS", "CAR_STATS",
    "IMAGENET_STATS", "CIFAR10_STATS", "CIFAR100_STATS", "ZEROSHOT_STATS",
    "build_bird200_discovery", "build_bird200_test",
    "build_dog120_discovery", "build_dog120_test",
    "build_flower102_discovery", "build_flower102_test",
    "build_pet37_discovery", "build_pet37_test",
    "build_car196_discovery", "build_car196_test",
]

DATA_STATS = {
    "bird": BIRD_STATS,
    "dog": DOG_STATS,
    "flower": FLOWER_STATS,
    "pet": PET_STATS,
    "car": CAR_STATS,
    "imagenet": IMAGENET_STATS,
    "cifar10": CIFAR10_STATS,
    "cifar100": CIFAR100_STATS,
    "zeroshot": ZERO_SHOT_TEMPLATES
}

DATA_DISCOVERY = {
    "bird": build_bird200_discovery,
    "dog": build_dog120_discovery,
    "flower": build_flower102_discovery,
    "pet": build_pet37_discovery,
    "car": build_car196_discovery,
}

DATA_GROUPING = {
    "bird": build_bird200_test,
    "dog": build_dog120_test,
    "flower": build_flower102_test,
    "pet": build_pet37_test,
    "car": build_car196_test,
}

DATA_TRANSFORM = {
    "bird": bird_transform,
    "dog": dog_transform,
    "flower": flower_transform,
    "pet": pet_transform,
    "car": car_transform,
}

DATA_AUGMENTATION = random_augmentation