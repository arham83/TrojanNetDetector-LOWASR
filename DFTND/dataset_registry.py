"""Central CIFAR-10/GTSRB/MNIST dataset selection used by every pipeline stage."""

from dataclasses import dataclass

from torchvision import datasets as tv_datasets, transforms

import dataset_input
import utilities
from robustness import datasets as robustness_datasets


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    num_classes: int
    image_size: int
    checkpoint_prefix: str
    numpy_dataset_class: type

    def make_robustness_dataset(self, data_path):
        dataset_classes = {
            'cifar10': robustness_datasets.CIFAR,
            'gtsrb': robustness_datasets.GTSRB,
            'mnist': robustness_datasets.MNIST,
        }
        return dataset_classes[self.name](data_path=data_path)

    def train_transform(self):
        if self.name == 'cifar10':
            return transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        if self.name == 'mnist':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        # Mirroring traffic signs can change their semantic meaning.
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ])

    def test_transform(self):
        steps = [transforms.Resize((self.image_size, self.image_size))]
        if self.name == 'mnist':
            steps.append(transforms.Grayscale(num_output_channels=1))
        steps.append(transforms.ToTensor())
        return transforms.Compose(steps)

    def make_torchvision_dataset(self, root, train, transform=None, download=False):
        transform = transform or (self.train_transform() if train else self.test_transform())
        if self.name == 'cifar10':
            return tv_datasets.CIFAR10(
                root=root, train=train, download=download, transform=transform
            )
        if self.name == 'gtsrb':
            return tv_datasets.GTSRB(
                root=root, split='train' if train else 'test',
                download=download, transform=transform
            )
        return tv_datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )


SPECS = {
    'cifar10': DatasetSpec(
        name='cifar10', display_name='CIFAR-10', num_classes=10,
        image_size=32, checkpoint_prefix='cifar',
        numpy_dataset_class=dataset_input.CIFAR10Data,
    ),
    'gtsrb': DatasetSpec(
        name='gtsrb', display_name='GTSRB', num_classes=43,
        image_size=32, checkpoint_prefix='gtsrb',
        numpy_dataset_class=dataset_input.GTSRB,
    ),
    'mnist': DatasetSpec(
        name='mnist', display_name='MNIST', num_classes=10,
        image_size=32, checkpoint_prefix='mnist',
        numpy_dataset_class=dataset_input.MNISTData,
    ),
}


def get_dataset_spec(name):
    return SPECS[utilities.normalize_dataset_name(name)]


def spec_from_config(config):
    return get_dataset_spec(config.data.dataset)


def checkpoint_path(config, role):
    if role not in config.model.checkpoints._fields:
        raise KeyError('Unknown checkpoint role {!r}'.format(role))
    return getattr(config.model.checkpoints, role)


def dataset_targets(dataset):
    """Return labels without decoding every image."""
    if hasattr(dataset, 'targets'):
        return list(dataset.targets)
    if hasattr(dataset, '_samples'):
        return [label for _, label in dataset._samples]
    if hasattr(dataset, 'samples'):
        return [label for _, label in dataset.samples]
    raise TypeError('Cannot extract labels from {}'.format(type(dataset).__name__))
