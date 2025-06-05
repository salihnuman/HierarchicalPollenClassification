import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import tifffile as tiff
import re
import random

cretan_root_dir = "/home/salih/DGRS_BigEarth/vs_code/Pollen/CretanPollen/Cropped Pollen Grains/Cropped Pollen Grains"
pollen73s_root_dir = "/home/salih/DGRS_BigEarth/vs_code/Pollen/Pollen73S"
pollen23e_root_dir = "/home/salih/DGRS_BigEarth/vs_code/Pollen/Pollen23E"
bingol_root_dir = "/home/salih/Desktop/DGRS_BigEarth/vs_code/Pollen/BingolPollen"
bingol_species_root_dir = "/home/salih/Desktop/DGRS_BigEarth/vs_code/Pollen/BingolPollen_Species"
main_root_dir = "/home/salih/DGRS_BigEarth/vs_code/Pollen"

class BingolPollenDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.name = 'BingolPollen'
        self._prepare_dataset()

    def _find_first_numeric_index(self, s):
        for i, char in enumerate(s):
            if char.isdigit():
                return i
        return -1

    def _prepare_dataset(self):
        print("Preparing dataset...")
        class_idx = 0
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                index = self._find_first_numeric_index(class_dir)
                if index != -1:
                    class_name = class_dir[:index]
                else:
                    raise ValueError(f"Invalid class directory name: {class_dir}")
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                    class_idx += 1
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
        print("Dataset preparation done.")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def name(self):
        return self.name


class BingolPollenDatasetHierarchical(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.species = []
        self.families = []
        self.gene = []
        self.species_to_idx = {}
        self.genus_to_idx = {}
        self.family_to_idx = {}
        self.gene_to_family = {}
        self.name = 'BingolPollen'
        self._get_gene_to_family_mapping()
        self._prepare_dataset()
        self.idx2species = {v: k for k, v in self.species_to_idx.items()}
        self.mapping = self._create_species_to_gene_family_mapping()

    def _get_gene_to_family_mapping(self):
        with open(os.path.join('gene_to_family.txt'), 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            self.gene_to_family[line.split("\t")[0]] = line.split("\t")[1]

    def _create_species_to_gene_family_mapping(self):
        species_to_gene_family = {}
        for species, idx in self.species_to_idx.items():
            genus = species.split(" ")[0]
            family = self.gene_to_family.get(genus, None)
            if family is not None:
                species_to_gene_family[species] = (genus, family)
        return species_to_gene_family

    def _prepare_dataset(self):
        print("Preparing dataset...")
        species_idx = 0
        genus_idx = 0
        family_idx = 0
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                species_name = class_dir
                genus_name = species_name.split(" ")[0]
                try:
                    family_name = self.gene_to_family[genus_name]
                except KeyError:
                    raise ValueError(f"Invalid genus name: {genus_name, species_name}")

                if species_name not in self.species_to_idx:
                    self.species_to_idx[species_name] = species_idx
                    species_idx += 1
                if genus_name not in self.genus_to_idx:
                    self.genus_to_idx[genus_name] = genus_idx
                    genus_idx += 1
                if family_name not in self.family_to_idx:
                    self.family_to_idx[family_name] = family_idx
                    family_idx += 1
                
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.families.append(self.family_to_idx[family_name])
                        self.species.append(self.species_to_idx[species_name])
                        self.gene.append(self.genus_to_idx[genus_name])
        print("Dataset preparation done.")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = (self.species[idx], self.gene[idx], self.families[idx])

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def name(self):
        return self.name
    
    def get_class_count(self):
        return len(self.species_to_idx), len(self.genus_to_idx), len(self.family_to_idx)

class Pollen73SDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.name = 'Pollen73S'
        self._prepare_dataset()

    def _prepare_dataset(self):
        class_idx = 0
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                class_name = class_dir
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                    class_idx += 1
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if img_path.endswith('.tif') or img_path.endswith('.TIF'):
            image = tiff.imread(img_path)[:, :, :3] # Load only RGB channels                  
            image = Image.fromarray(image)
        else:
            image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
class Pollen23EDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.name = 'Pollen23E'
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        class_idx = 0
        for filename in os.listdir(self.root_dir):
            class_name = re.split(r'[_ ]', filename)[0]
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = class_idx
                class_idx += 1
            img_path = os.path.join(self.root_dir, filename)
            if os.path.isfile(img_path):
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class CretanPollenDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.name = 'CretanPollen'
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        class_idx = 0
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                class_name = class_dir.split('.')[1]
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                    class_idx += 1
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

class CombinedPollenDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.datasets = []
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self._prepare_combined_dataset()

    def _prepare_combined_dataset(self):
        dataset_classes = [Pollen73SDataset, Pollen23EDataset, CretanPollenDataset]
        dir_names = ['/Pollen73S', '/Pollen23E', '/CretanPollen/Cropped Pollen Grains/Cropped Pollen Grains']
        for i, dataset_class in enumerate(dataset_classes):
            dataset = dataset_class(self.root_dir + dir_names[i], transform=self.transform)
            self.datasets.append(dataset)
            self.image_paths.extend(dataset.image_paths)
            self.labels.extend(dataset.labels)
            for class_name, class_idx in dataset.class_to_idx.items():
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if img_path.endswith('.tif') or img_path.endswith('.TIF'):
            image = tiff.imread(img_path)[:, :, :3] # Load only RGB channels                  
            image = Image.fromarray(image)
        else:
            image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def show_example_images(dataset):
    sample = random.sample(range(len(dataset)), 6)
    example_images = [dataset[i][0] for i in sample]
    example_targets = [dataset[i][1] for i in sample]

    # Show the example images using PIL
    for i, img in enumerate(example_images):
        img_pil = transforms.ToPILImage()(img)
        img_pil.show(title=f"Image {i} label: {example_targets[i]}")
        img_pil.save(f"example_image_{i}.png")

    # Print the label names for each example image
    for i, target in enumerate(example_targets):
        label = [k for k, v in dataset.class_to_idx.items() if v == target]
        print(f"Image {i} label: {', '.join(label)}")

def calculate_dataset_stats(dataset):
    # Calculate mean and std for each channel
    mean = np.zeros(3)
    std = np.zeros(3)
    num_samples = 0
    for i in range(len(dataset)):
        img, _ = dataset[i]
        mean += np.mean(img.numpy(), axis=(1, 2))
        std += np.std(img.numpy(), axis=(1, 2))
        num_samples += 1
    mean /= num_samples
    std /= num_samples
    print(f"Mean: {np.round(mean, 3)}")
    print(f"Std: {np.round(std, 3)}")

def print_common_labels(dataset1, dataset2):
    labels1 = set(dataset1.class_to_idx.keys())
    labels2 = set(dataset2.class_to_idx.keys())
    common_labels = []
    for label2 in labels2:
        for label in labels1:
            if label2.lower() in label.lower():
                common_labels.append(label2)
    print(f"Dataset 1 labels: {labels1}")
    print(f"\nDataset 2 labels: {labels2}")
    print(f"\nCommon labels: {common_labels}")

# Example usage
if __name__ == "__main__":
    root_dir = bingol_root_dir
    #root_dir2 = cretan_root_dir
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = BingolPollenDataset(root_dir, transform=transform)
    #dataset2 = CretanPollenDataset(root_dir2, transform=transform)
    #print(f"Number of samples: {len(dataset)}")
    #print("Number of classes:", dataset.get_class_count())
    #calculate_dataset_stats(dataset)
    #print_common_labels(dataset, dataset2)
    show_example_images(dataset)
    #print(dataset.genus_to_idx.keys())