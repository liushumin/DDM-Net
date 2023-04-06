import sys
sys.path.append('../')
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, ToPILImage
from .dataset_PPI import DatasetFromFolder_PPI
from .dataset import DatasetFromFolder

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform():
    return Compose([
        # ToPILImage(),
        # CenterCrop(crop_size),
        # Resize(crop_size // upscale_factor),
        ToTensor(),
    ])

def target_transform( ):
    return Compose([
        # CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set_opt(dir, upscale_factor,norm_flag,augment_flag):
    return DatasetFromFolder(dir,norm_flag,
                             input_transform=input_transform(),
                             target_transform=target_transform(),
                             augment = augment_flag
                             )

def get_PPI_training_set_opt(dir, upscale_factor,norm_flag,augment_flag):
    return DatasetFromFolder_PPI(dir,norm_flag,
                             input_transform=input_transform(),
                             target_transform=target_transform(),
                             augment = augment_flag
                             )


def get_test_set_opt(dir, upscale_factor, norm_flag):
    return DatasetFromFolder(dir,norm_flag,
                             input_transform=input_transform(),
                             target_transform=target_transform()
                             )
def get_PPI_test_set_opt(dir, upscale_factor,norm_flag):
    return DatasetFromFolder_PPI(dir,norm_flag,
                             input_transform=input_transform(),
                             target_transform=target_transform()
                             )