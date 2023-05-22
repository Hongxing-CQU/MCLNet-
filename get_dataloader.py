import torchvision
from torch.utils.data import DataLoader

from ModelNet40 import ModelNet40, pc_normalize
from transforms import get_transforms


def get_modelnet40_dataloader(noise_type: str,
                              root: str,
                              rot_mag: float = 45.0,
                              trans_mag: float = 0.5,
                              num_points: int = 1024,
                              partial_p_keep: list = None,
                              unseen: bool = False,
                              batch_size: int = 8,
                              test_batch_size = 4
                              ):
    if partial_p_keep is None:
        partial_p_keep = [0.7, 0.7]
    train_transforms, test_transforms = get_transforms(noise_type, rot_mag, trans_mag, num_points, partial_p_keep)
    train_transforms = torchvision.transforms.Compose(train_transforms)
    test_transforms = torchvision.transforms.Compose(test_transforms)

    train_loader = DataLoader(
        ModelNet40(root=root, partition='train', unseen=unseen, num_points=num_points, transforms=train_transforms),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )

    test_loader = DataLoader(
        ModelNet40(root=root, partition='test', unseen=False, num_points=1024, transforms=test_transforms),
        batch_size=test_batch_size,shuffle=False,drop_last=False,num_workers=4
    )
    return train_loader, test_loader
