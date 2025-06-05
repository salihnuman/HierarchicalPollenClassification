from dataset import BingolPollenDataset, BingolPollenDatasetHierarchical, Pollen73SDataset, CombinedPollenDataset, Pollen23EDataset, CretanPollenDataset
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from losses import FocalLoss, InfoNCE, SimCLRLoss, HierarchicalCrossEntropyLoss, AutomaticWeightedLoss
from utils import TwoCropTransform
import numpy as np
import random
from sklearn.metrics import f1_score, matthews_corrcoef
import wandb
#import torchmetrics.functional as metrics
from torch.backends import cudnn
from models import HierarchicalPollenModel, PollenModel, Classifier
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
#from transformers import AutoModel

num_classes = {
    'Pollen73S': 73,
    'Pollen23E': 23,
    'CretanPollen': 20,
    'BingolPollen': 47,
    'BingolPollenHierarchical': 47,
    'CombinedPollen': 116
}

datasets = {
    'Pollen73S': Pollen73SDataset,
    'Pollen23E': Pollen23EDataset,
    'CretanPollen': CretanPollenDataset,
    'BingolPollen': BingolPollenDataset,
    'BingolPollenHierarchical': BingolPollenDatasetHierarchical,
    'CombinedPollen': CombinedPollenDataset
}


TRAIN_MODE = True
HIERARCHY = 3   # 1 for no hierarchy, 2 for familia and species, 3 for familia, genus and species
BACKBONE = 'swin'  # resnet50, xlstm, swin
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.1
MOMENTUM = 0.9
SEED = 42
DATASET = 'BingolPollenHierarchical'   # Pollen73S, Pollen23E, CretanPollen or BingolPollen (for transformations)
NUM_CLASSES = num_classes[DATASET]        # 73 for Pollen73S, 23 for Pollen23E, 20 for CretanPollen, 47 for ours
ROOT_DIR = '/home/salih/Desktop/DGRS_BigEarth/vs_code/Pollen/BingolPollen_Species'
LOG_INTERVAL = 50

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def test(model, dataloader, device, dataset, level = "species"):
    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for images, (labels_species, labels_genus, labels_familia) in dataloader:
            if level == "species":
                labels = labels_species
            elif level == "familia":
                labels = labels_familia

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if HIERARCHY > 1:
                _, predicted = torch.max(outputs[level], 1)
            else:
                _, predicted = torch.max(outputs, 1)
                if level == "familia":
                    # Get familia preds from dataset mapping
                    species_to_gene_family = dataset.mapping
                    idx_to_species = dataset.idx2species
                    family2idx = dataset.family_to_idx
                    predicted = [family2idx[species_to_gene_family[idx_to_species[pred.item()]][1]] for pred in predicted]
                    predicted = torch.tensor(predicted, device=device)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu())
            all_labels.extend(labels.cpu())
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='macro')
        mcc = matthews_corrcoef(all_labels, all_preds)

        print(f"Test accuracy: {accuracy}")
        print(f"Test F1: {f1}")
        print(f"Test MCC: {mcc}")

def freeze_backbone(model):
    """
    Freezes the backbone of a HierarchicalPollenModel so only the classifier is trainable.
    """
    # Assuming your model has an attribute called 'backbone'
    for param in model.backbone.parameters():
        param.requires_grad = False

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    print(f"Hierarchy: {HIERARCHY}")
    print(f"Backbone: {BACKBONE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    setSeed(SEED)
    root_dir = ROOT_DIR

    mean_stds = {
        'Pollen73S': ([0.584, 0.522, 0.583], [0.1, 0.14, 0.112]),
        'Pollen23E': ([0.537, 0.554, 0.58], [0.127, 0.14, 0.174]),
        'CretanPollen': ([0.609, 0.431, 0.531], [0.076, 0.158, 0.098]),
        'BingolPollen': ([0.774, 0.702, 0.761], [0.058, 0.162, 0.068]),
        'CombinedPollen': ([0.609, 0.431, 0.531], [0.076, 0.158, 0.098]),
        'BingolPollenHierarchical': ([0.774, 0.702, 0.761], [0.058, 0.162, 0.068])}

    
    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
        #transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_stds[DATASET][0], std=mean_stds[DATASET][1]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_stds[DATASET][0], std=mean_stds[DATASET][1]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_stds[DATASET][0], std=mean_stds[DATASET][1]),
    ])


    dataset = datasets[DATASET](root_dir, transform=val_transform)  # train_transform
    
    
    # Extract labels for stratified splitting
    labels = [dataset[i][1][0] for i in range(len(dataset))]

    # Define stratified splitter
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)

    # Perform the split
    for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels):
        train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(train_idx, np.array(labels)[train_idx]))

    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Apply different transforms to each subset
    #train_dataset.dataset.transform = TwoCropTransform(train_transform)
    #val_dataset.dataset.transform = val_transform
    #test_dataset.dataset.transform = test_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # logger - start a new wandb run to track this script
    if TRAIN_MODE:
        wandb.login()
        wandb.init(
            project="Pollen",
            config={
            "learning_rate": LR,
            "architecture": "Resnet50",
            "dataset": "Pollen73S",                         # Pollen73S, Pollen23E, CretanPollen or BingolPollen 
            "epochs": EPOCHS,
            }
        )

    if TRAIN_MODE:
        if HIERARCHY > 1:
            # Experiment on different loss functions and lr
            lrs = [0.0005]
            loss_functions = [torch.nn.CrossEntropyLoss()]      #, FocalLoss(task_type='multi-class', num_classes=NUM_CLASSES)
            best_lr = None
            best_loss_func = None
            best_f1 = 0
            for criterion in loss_functions:
                for lr in lrs:
                    print(f"Training with {criterion} and lr={lr}")
                    # Create model and optimizer
                    #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)                        # pretrained ResNet-50 model
                    model = HierarchicalPollenModel(backbone_architecture=BACKBONE, pretrained=True, hierarchy_level = HIERARCHY, freeze_early_layers=1)  # HierarchicalPollenModel
                    
                    model.to(device)
                    #awl = AutomaticWeightedLoss(num=3).to(device)
                    #optimizer = torch.optim.Adam((list(model.parameters()) + list(awl.parameters())), lr=lr)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)
                    
                    best_epoch_f1 = 0
                    best_epoch_model = None
                    best_epoch = -1
                    
                    if HIERARCHY == 2:
                        for epoch in range(EPOCHS):
                            model.train()
                            for images, (labels_species, _, labels_familia) in train_loader:
                                images = images.to(device)

                                labels_familia = labels_familia.to(device)
                                labels_species = labels_species.to(device)

                                optimizer.zero_grad()
                                outputs = model(images)
                                assert outputs['familia'].shape[0] == labels_familia.shape[0], f"Output shape {outputs['familia'].shape} does not match labels shape {labels_familia.shape}"
                                assert outputs['species'].shape[0] == labels_species.shape[0], f"Output shape {outputs['species'].shape} does not match labels shape {labels_species.shape}"
                                
                                
                                loss_familia = criterion(outputs['familia'], labels_familia)
                                loss_species = criterion(outputs['species'], labels_species)
                                loss = 0.8 * loss_species + 0.2 * loss_familia                              # Maybe use automatic weighted loss
                                
                                loss.backward()
                                optimizer.step()
                            scheduler.step()

                            # Validation phase
                            model.eval()
                            val_preds = []
                            val_labels = []
                            with torch.no_grad():
                                correct = 0
                                total = 0
                                for images, (labels_species, _, labels_familia) in val_loader:
                                    labels = labels_species
                                    images = images.to(device)
                                    labels = labels.to(device)
                                    outputs = model(images)
                                    _, predicted = torch.max(outputs['species'], 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()
                                    val_preds.extend(predicted.cpu())
                                    val_labels.extend(labels.cpu())
                                accuracy = correct / total
                                f1 = f1_score(val_labels, val_preds, average='macro')
                                
                                wandb.log({f'training loss {lr}, {criterion}': loss.item()})
                                print(f"Epoch {epoch}: Loss: {loss.item()}")
                                print(f"Epoch {epoch}: Validation accuracy: {accuracy}")
                                print(f"Epoch {epoch}: Validation F1: {f1}")
                            
                            if f1 > best_epoch_f1:
                                best_epoch_f1 = f1
                                best_epoch_accuracy = accuracy
                                best_epoch_model = model.state_dict()
                                best_epoch = epoch

                        print(f"Best F1 for lr={lr}, criterion={criterion}: {best_epoch_f1} at epoch {best_epoch}")

                        with open('results.txt', 'a') as f:
                            f.write(f"Loss: {criterion}, lr: {lr}, F1: {best_epoch_f1}, Accuracy: {best_epoch_accuracy}\n")

                        if best_epoch_f1 > best_f1:
                            best_f1 = best_epoch_f1
                            best_loss_fun = criterion
                            best_lr = lr
                            best_model = best_epoch_model


                    elif HIERARCHY == 3:
                        for epoch in range(EPOCHS):
                            model.train()
                            for images, (labels_species, labels_genus , labels_familia) in train_loader:
                                images = images.to(device)

                                labels_familia = labels_familia.to(device)
                                labels_genus = labels_genus.to(device)
                                labels_species = labels_species.to(device)

                                optimizer.zero_grad()
                                outputs = model(images)
                                assert outputs['familia'].shape[0] == labels_familia.shape[0], f"Output shape {outputs['familia'].shape} does not match labels shape {labels_familia.shape}"
                                assert outputs['genus'].shape[0] == labels_genus.shape[0], f"Output shape {outputs['genus'].shape} does not match labels shape {labels_genus.shape}"
                                assert outputs['species'].shape[0] == labels_species.shape[0], f"Output shape {outputs['species'].shape} does not match labels shape {labels_species.shape}"
                                
                                
                                loss_familia = criterion(outputs['familia'], labels_familia)
                                loss_species = criterion(outputs['species'], labels_species)
                                loss_genus = criterion(outputs['genus'], labels_genus)
                                loss = 0.5 * loss_species + 0.3 * loss_genus + 0.2 * loss_familia                              # Maybe use automatic weighted loss
                                #loss = awl(loss_species, loss_genus, loss_familia)
                                loss.backward()
                                optimizer.step()

                            model.eval()
                            val_preds = []
                            val_labels = []
                            with torch.no_grad():
                                correct = 0
                                total = 0
                                for images, (labels_species, labels_genus, labels_familia) in val_loader:
                                    labels = labels_species
                                    images = images.to(device)
                                    labels = labels.to(device)
                                    outputs = model(images)
                                    _, predicted = torch.max(outputs['species'], 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()
                                    val_preds.extend(predicted.cpu())
                                    val_labels.extend(labels.cpu())
                                accuracy = correct / total
                                f1 = f1_score(val_labels, val_preds, average='macro')
                                
                                wandb.log({f'training loss {lr}, {criterion}': loss.item()})
                                print(f"Epoch {epoch}: Loss: {loss.item()}")
                                print(f"Epoch {epoch}: Validation accuracy: {accuracy}")
                                print(f"Epoch {epoch}: Validation F1: {f1}")
                            
                            if f1 > best_epoch_f1:
                                best_epoch_f1 = f1
                                best_epoch_accuracy = accuracy
                                best_epoch_model = model.state_dict()
                                best_epoch = epoch

                        print(f"Best F1 for lr={lr}, criterion={criterion}: {best_epoch_f1} at epoch {best_epoch}")
                        
                        with open('results.txt', 'a') as f:
                            f.write(f"Loss: {criterion}, lr: {lr}, F1: {best_epoch_f1}, Accuracy: {best_epoch_accuracy}\n")

                        if best_epoch_f1 > best_f1:
                            best_f1 = best_epoch_f1
                            best_loss_fun = criterion
                            best_lr = lr
                            best_model = best_epoch_model

            # save the best model
            print(f"Best F1: {best_f1} with loss: {best_loss_fun} and lr: {best_lr}")
            torch.save(best_model, f"experiments/{DATASET}/{BACKBONE}_{best_loss_fun}_lr{best_lr}_{BATCH_SIZE}batch_{EPOCHS}epoch_{HIERARCHY}hierarchy_weighted532_freezeearly.pt")
            print(f"Model saved to experiments/{DATASET}/{BACKBONE}_{best_loss_fun}_lr{best_lr}_{BATCH_SIZE}batch_{EPOCHS}epoch_{HIERARCHY}hierarchy_weighted532_freezeearly.pt")
        else:
            lrs = [0.0005]
            loss_functions = [torch.nn.CrossEntropyLoss()]      #, FocalLoss(task_type='multi-class', num_classes=NUM_CLASSES)
            #loss_functions = [HierarchicalCrossEntropyLoss(num_classes=247, dataset=dataset)]
            best_lr = None
            best_loss_func = None
            best_f1 = 0
            for criterion in loss_functions:
                for lr in lrs:
                    print(f"Training with {criterion} and lr={lr}")
                    # Create model and optimizer
                    #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)                        # pretrained ResNet-50 model
                    model = PollenModel(backbone_architecture=BACKBONE, pretrained=True, stage="species", freeze_early_layers=1)  # PollenModel

                    #model = torch.load('experiments/BingolPollenHierarchical/contrastive/resnet50_0.03_300epoch_hierarchicalcontrastive_finertocoarser.pt', map_location="cpu", weights_only=False)
                    #model.load_state_dict(state_dict)
                    #model.classifier = Classifier(in_features=model.backbone.out_features, out_features=274)

                    #freeze_backbone(model)
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)

                    best_epoch_f1 = 0
                    best_epoch_model = None
                    best_epoch = -1
                    
                    for epoch in range(EPOCHS):
                        model.train()
                        #model.backbone.eval()
                        #model.classifier.train()
                        for images, (labels_species, _ , labels_familia) in train_loader:
                            images = images.to(device)
                            labels_species = labels_species.to(device)
                            optimizer.zero_grad()
                            outputs = model(images)
                            #assert outputs.shape[0] == labels_species.shape[0], f"Output shape {outputs.shape} does not match labels shape {labels_species.shape}"
                            #assert outputs.shape[1] == NUM_CLASSES, f"Output shape {outputs.shape} does not match num_classes {NUM_CLASSES}"
                            
                            loss = criterion(outputs, labels_species)
                            
                            loss.backward()
                            """
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    print(f"{name}: {param.grad.abs().mean().item()}")          # To check gradient flow
                                else:
                                    print(f"{name}: No gradient")
                            """
                            optimizer.step()

                        scheduler.step()
                        model.eval()
                        val_preds = []
                        val_labels = []
                        with torch.no_grad():
                            correct = 0
                            total = 0
                            for images, (labels_species, _, labels_familia) in val_loader:
                                labels = labels_species
                                images = images.to(device)
                                labels = labels.to(device)
                                outputs = model(images)
                                _, predicted = torch.max(outputs, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                                val_preds.extend(predicted.cpu())
                                val_labels.extend(labels.cpu())
                            accuracy = correct / total
                            f1 = f1_score(val_labels, val_preds, average='macro')
                            
                            wandb.log({f'training loss {lr}, {criterion}': loss.item()})
                            print(f"Epoch {epoch}: Loss: {loss.item()}")
                            print(f"Epoch {epoch}: Validation accuracy: {accuracy}")
                            print(f"Epoch {epoch}: Validation F1: {f1}")
                        
                        if f1 > best_epoch_f1:
                            best_epoch_f1 = f1
                            best_epoch_accuracy = accuracy
                            best_epoch_model = model.state_dict()
                            best_epoch = epoch
                    
                    print(f"Best F1 for lr={lr}, criterion={criterion}: {best_epoch_f1} at epoch {best_epoch}")

                    with open('results.txt', 'a') as f:
                        f.write(f"Loss: {criterion}, lr: {lr}, F1: {best_epoch_f1}, Accuracy: {best_epoch_accuracy}\n")

                    if best_epoch_f1 > best_f1:
                        best_f1 = best_epoch_f1
                        best_loss_fun = criterion
                        best_lr = lr
                        best_model = best_epoch_model

            print(f"Best F1: {best_f1} with loss: {best_loss_fun} and lr: {best_lr}")

            # save the best model
            torch.save(best_model, f"experiments/{DATASET}/{BACKBONE}_{best_loss_fun}_lr{best_lr}_{BATCH_SIZE}batch_{EPOCHS}epoch_nohierarchy_freezeearly.pt")
            print(f"Model saved to experiments/{DATASET}/{BACKBONE}_{best_loss_fun}_lr{best_lr}_{BATCH_SIZE}batch_{EPOCHS}epoch_nohierarchy_freezeearly.pt")
    else:
        # Load the model from local
        
        if HIERARCHY > 1:
            model = HierarchicalPollenModel(backbone_architecture=BACKBONE, pretrained=True, hierarchy_level = HIERARCHY)
        else:
            model = PollenModel(backbone_architecture=BACKBONE, pretrained=True, stage="species")
        state_dict = torch.load('experiments/BingolPollenHierarchical/xlstm_CrossEntropyLoss()_lr0.0005_16batch_100epoch_nohierarchy_freezeearly.pt', map_location="cpu")
        
        model.load_state_dict(state_dict)
        model.to(device)

        # TEST_MODE with whatever is the best validation model
        print("Testing the model on class level")
        test(model, test_loader, device, dataset)
        print("Testing the model on family level")
        test(model, test_loader, device, dataset, level="familia")