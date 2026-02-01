import os
from PIL import Image
import scipy
from scipy import io as mat_io
import scipy.misc
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import list_dir
from torch.utils.data import DataLoader
from data.data_stats import DOG_STATS
import pathlib
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


class DogDiscovery120:
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
                self.samples.append(os.path.join(folder, name))
                self.targets.append(label)

        self.classes = DOG_STATS['class_names']
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


class DogDataset(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    def __init__(self, root, train=True, transform=None):

        # self.root = join(os.path.expanduser(root), self.folder)
        self.root = root
        self.train = train
        self.transform = transform

        split = self.load_split()

        self.images_folder = os.path.join(self.root, 'Images')
        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)


        self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]
        self._flat_breed_images = self._breed_images

        # Class names and target index is aligned!
        self.data = [img_path for img_path, _ in self._flat_breed_images]
        self.target = [label for _, label in self._flat_breed_images]
        self.classes = DOG_STATS['class_names']

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # return image, target_class, index
        return image, target_class, image_path  # just for visualization

    def load_split(self):
        if self.train:
            split = mat_io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = mat_io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        else:
            split = mat_io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = mat_io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))

        return counts


def build_dog120_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = DogDiscovery120(cfg['data_dir'], folder_suffix=folder_suffix)
    return set_to_discover


def build_dog120_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = DogDataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def create_discovery_set(root, files, targets, class_names, num_per_category=3, selection='largest'):
    # selection = 'random' or 'largest'
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
            dest_img = file_template.format(str(label).zfill(3), cleaned_names[label], src_img.split('/')[-1])
            destination_path = os.path.join(destination_folder, dest_img)
            shutil.copyfile(src_img, destination_path)
            num_copied += 1
            print(f"[{num_copied}]  copied <{destination_path}> from <{src_img}>")


def generate_long_tail_distribution(num_categories=120):
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
        10, 10, 10, 10, 10, 10, 10,  9,  8,  7,  7,  6,  6,  5,  5,  5,  5,
        4,  4,  4,  4,  4,  4,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1
    ]

    # selection = 'random' or 'largest'
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
            dest_img = file_template.format(str(label).zfill(3), cleaned_names[label], src_img.split('/')[-1])
            destination_path = os.path.join(destination_folder, dest_img)
            shutil.copyfile(src_img, destination_path)
            num_copied += 1
            print(f"[{num_copied}]  copied <{destination_path}> from <{src_img}>")


if __name__ == "__main__":
    root = "<your_path>"
    tfms = _transform(224)

    dataset = DogDataset(root, train=True, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    print(f'Num All Classes: {len(set(dataset.target))}')
    print(f"Num All Images: {len(dataset.data)}")

    print(f'Len set: {len(dataset)}')
    print(f"Image {dataset.data[-1]} has Label {dataset.target[-1]} whose class name {dataset.classes[dataset.target[-1]]}")

    # print("Create random unlabeled set as wild discovery set")
    # num_per_category_list = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    # # num_per_category_list = [3]
    # for num_per_category in num_per_category_list:
    #     root_disco = os.path.join(root, f"images_discovery_all_{num_per_category}")
    #
    #     if not os.path.exists(root_disco):
    #         os.makedirs(root_disco)
    #
    #     sample_root = os.path.join(root, "Images")
    #     sample_paths = [os.path.join(sample_root, p) for p in dataset.data]
    #
    #     create_discovery_set(root_disco, sample_paths, dataset.target, dataset.classes,
    #                          num_per_category=num_per_category, selection='largest')

    # print("Create long-tailed unlabeled set as wild discovery set")
    # root_disco = os.path.join(root, f"images_discovery_all_random")
    #
    # if not os.path.exists(root_disco):
    #     os.makedirs(root_disco)
    # sample_root = os.path.join(root, "Images")
    # sample_paths = [os.path.join(sample_root, p) for p in dataset.data]
    # create_long_tail_discovery_set(
    #     root_disco, sample_paths, dataset.target, dataset.classes, selection='largest'
    # )
