import os
from PIL import Image
import torchvision.transforms as transforms
import pathlib
from typing import Any, Callable, Optional, Union, Tuple
from typing import Sequence
import numpy as np
from scipy import io as mat_io
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import PET_STATS
import random
import shutil
from copy import deepcopy


def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # ImageNet
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def make_dataset(dir, image_ids, targets):
    assert (len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


class PetDiscovery37:
    def __init__(self, root, folder_suffix=''):
        img_root = os.path.join(root, f'images_discovery_all{folder_suffix}')

        self.class_folders = os.listdir(img_root)  # 100 x 1
        for i in range(len(self.class_folders)):
            self.class_folders[i] = os.path.join(img_root, self.class_folders[i])

        self.samples = []
        self.targets = []
        for folder in self.class_folders:
            label = int(folder.split('/')[-1][:3])
            # label = label - 1
            file_names = os.listdir(folder)

            for name in file_names:
                self.targets.append(label)
                self.samples.append(os.path.join(folder, name))

        self.classes = PET_STATS['class_names']
        self.index = 0

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.samples):
            raise StopIteration
        img = Image.open(self.samples[self.index]).convert("RGB")
        target = self.targets[self.index]
        img_path = self.samples[self.index]
        self.index += 1
        return img, target, img_path


class PetDataset(Dataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    valid_target_types = ("category", "segmentation")
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(
            self,
            root: str,
            train: bool = False,
            transform: Optional[Callable] = None,
            split: str = "test",
            target_types: Union[Sequence[str], str] = "category",
            loader=default_loader
    ):
        if train:
            split = "trainval"
        else:
            split = "test"

        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.split = split

        if isinstance(target_types, str):
            target_types = [target_types]

        self.root = root
        self.transform = transform
        self.loader = loader

        self._base_folder = pathlib.Path(self.root)
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self.split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        samples = make_dataset(self.root, image_ids, self._labels)
        self.samples = samples
        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self.uq_idxs = np.array(range(len(self)))

        self.data = [path for path, label in self.samples]
        self.target = [label for path, label in self.samples]
        self.classes = PET_STATS['class_names']

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[idx]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, target, self.uq_idxs[idx]
        return sample, target, path     # just for visualization


def build_pet37_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = PetDiscovery37(cfg['data_dir'], folder_suffix=folder_suffix)
    return set_to_discover


def build_pet37_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = PetDataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def create_discovery_set(root, files, targets, class_names, num_per_category=3, selection='largest'):
    target_image_dict = {i: [] for i in range(len(class_names))}
    cleaned_names = deepcopy(class_names)
    cleaned_names = [name.replace(' ', '_') for name in cleaned_names]
    cleaned_names = [name.replace('-', '_') for name in cleaned_names]

    folder_template = "{}.{}"
    file_template = "{}.{}_{}"

    for label, image in zip(targets, files):
        target_image_dict[label].append(image)

    num_copied = 0
    for label, images in target_image_dict.items():
        image_paths = deepcopy(images)
        random.shuffle(image_paths)

        if selection == 'random':
            random.shuffle(image_paths)
        else:
            image_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
            image_paths = image_paths[:50]
            random.shuffle(image_paths)

        selected_images = image_paths[:num_per_category]

        destination_folder = folder_template.format(str(label).zfill(3), cleaned_names[label])
        destination_folder = os.path.join(root, destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for src_img in selected_images:
            dest_img = file_template.format(str(label).zfill(3), cleaned_names[label], src_img.split('_')[-1])
            destination_path = os.path.join(destination_folder, dest_img)
            shutil.copyfile(src_img, destination_path)
            num_copied += 1
            print(f"[{num_copied}]  copied <{destination_path}> from <{src_img}>")


def generate_long_tail_distribution(num_categories=37):
    import numpy as np
    import matplotlib.pyplot as plt

    categories = num_categories

    # Generate power-law distributed random numbers.
    # The parameter 'a' controls the shape of the distribution.
    # Smaller 'a' gives a heavier tail.
    a = 2.0
    samples = np.random.zipf(a, categories)

    # Adjust samples to ensure they're within the range 1 to 10
    adjusted_samples = np.clip(samples, 1, 10)

    # Sort samples in descending order
    sorted_samples = np.sort(adjusted_samples)[::-1]

    print(sorted_samples)

    # Optional: Plot to see the distribution
    plt.bar(range(categories), sorted_samples)
    plt.xlabel('Categories')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images across Categories')
    plt.show()


def create_long_tail_discovery_set(root, files, targets, class_names, selection='largest'):
    distribution = [
        10, 10, 10, 10, 10, 10,  8,  5,  4,  3,  2,  2,  2,  2,  2,  2,  2,
        2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1
    ]

    target_image_dict = {i: [] for i in range(len(class_names))}
    cleaned_names = deepcopy(class_names)
    cleaned_names = [name.replace(' ', '_') for name in cleaned_names]
    cleaned_names = [name.replace('-', '_') for name in cleaned_names]

    folder_template = "{}.{}"
    file_template = "{}.{}_{}"

    for label, image in zip(targets, files):
        target_image_dict[label].append(image)

    num_copied = 0
    for label, images in target_image_dict.items():
        num_per_category = distribution[label]

        image_paths = deepcopy(images)
        random.shuffle(image_paths)

        if selection == 'random':
            random.shuffle(image_paths)
        else:
            image_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
            image_paths = image_paths[:50]
            random.shuffle(image_paths)

        selected_images = image_paths[:num_per_category]

        destination_folder = folder_template.format(str(label).zfill(3), cleaned_names[label])
        destination_folder = os.path.join(root, destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for src_img in selected_images:
            dest_img = file_template.format(str(label).zfill(3), cleaned_names[label], src_img.split('_')[-1])
            destination_path = os.path.join(destination_folder, dest_img)
            shutil.copyfile(src_img, destination_path)
            num_copied += 1
            print(f"[{num_copied}]  copied <{destination_path}> from <{src_img}>")


if __name__ == "__main__":
    root = "<your_path>"
    tfms = _transform(224)

    dataset = PetDataset(root, train=True, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    print(f'Num All Classes: {len(set(dataset.target))}')
    print(f"Num All Images: {len(dataset.data)}")

    print(f'Len set: {len(dataset)}')
    test_idx = 137
    print(f"Image {dataset.data[test_idx]} has Label {dataset.target[test_idx]} whose class name is {dataset.classes[dataset.target[test_idx]]}")

    # print("Create random unlabeled set as wild discovery set")
    # num_per_category_list = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    # # num_per_category_list = [3]
    #
    # for num_per_category in num_per_category_list:
    #     root_disco = os.path.join(root, f"images_discovery_all_{num_per_category}")
    #
    #     if not os.path.exists(root_disco):
    #         os.makedirs(root_disco)
    #
    #     create_discovery_set(root_disco, dataset.data, dataset.target, dataset.classes,
    #                          num_per_category=num_per_category, selection='largest')

    # print("Create long-tailed unlabeled set as wild discovery set")
    # root_disco = os.path.join(root, f"images_discovery_all_random")
    #
    # if not os.path.exists(root_disco):
    #     os.makedirs(root_disco)
    #
    # create_long_tail_discovery_set(
    #     root_disco, dataset.data, dataset.target, dataset.classes, selection='largest'
    # )
