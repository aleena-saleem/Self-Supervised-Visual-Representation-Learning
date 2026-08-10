
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SimCLRTransform:

    def __init__(self, image_size=96):

        self.transform = transforms.Compose([

            transforms.RandomResizedCrop(
                image_size,
                scale=(0.2, 1.0)
            ),

            transforms.RandomHorizontalFlip(),

            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.8,
                        contrast=0.8,
                        saturation=0.8,
                        hue=0.2
                    )
                ],
                p=0.8
            ),

            transforms.RandomGrayscale(p=0.2),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, image):

        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2


class SimCLRDataset(Dataset):

    def __init__(
        self,
        dataset,
        transform=None
    ):

        self.dataset = dataset
        self.transform = transform

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        image, _ = self.dataset[index]

        if self.transform is not None:

            view1, view2 = self.transform(image)

        else:

            view1 = image
            view2 = image

        return view1, view2
