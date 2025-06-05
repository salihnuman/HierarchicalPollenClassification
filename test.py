import torch
import torch.nn.functional as F
from dataset import BingolPollenDatasetHierarchical
from models import PollenModel
from losses import HierarchicalCrossEntropyLoss
from torchvision import transforms
from torch.utils.data import DataLoader

root_dir = "/home/salih/Desktop/DGRS_BigEarth/vs_code/Pollen/BingolPollen_Species"
model = PollenModel(backbone_architecture="resnet50", pretrained=True, stage="species")
model = model.to("cuda")
val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
dataset = BingolPollenDatasetHierarchical(root_dir, transform=val_transform)
criterion = HierarchicalCrossEntropyLoss(274, dataset)

dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
for i, (images, (species, genus, family)) in enumerate(dataloader):
    images = images.to(device)
    species = species.to(device)
    genus = genus.to(device)
    family = family.to(device)

    # Forward pass
    outputs = model(images)
    # Compute loss
    loss = criterion(outputs, species)
    print(f"Loss: {loss.item()}")

    """
    for label in species:
        label = label.item()
        print(criterion.idx_to_species[label])
        print(criterion.getGenus(label), criterion.getFamily(label))
    """

    raise ValueError("Stop here")


    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item()}")