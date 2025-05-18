from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .wrapper import WrapperDataset
from .imagenet_a import ImageNet_a
from .imagenet_r import ImageNet_r
from .imagenet_s import ImageNet_s
from .imagenetv2 import ImageNetv2

dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech-101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                "imagenet_a": ImageNet_a,
                "imagenet_r": ImageNet_r,
                "imagenet_s": ImageNet_s,
                "imagenetv2": ImageNetv2
                }


def build_dataset(dataset, root, num_shots, subsample, transform=None, type='train', seed=0, imb_domain = 'base'):
    return dataset_list[dataset](root, num_shots, subsample, transform=transform, type=type, seed=seed, imb_domain=imb_domain)