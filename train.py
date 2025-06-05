from dataset import BingolPollenDataset, Pollen73SDataset, CombinedPollenDataset, Pollen23EDataset, CretanPollenDataset
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from losses import FocalLoss, InfoNCE, SimCLRLoss
from utils import TwoCropTransform
import numpy as np
import random
from sklearn.metrics import f1_score
import wandb
import torchmetrics.functional as metrics
from torch.backends import cudnn
#from transformers import AutoModel

num_classes = {
    'Pollen73S': 73,
    'Pollen23E': 23,
    'CretanPollen': 20,
    'BingolPollen': 47,
    'CombinedPollen': 116
}

datasets = {
    'Pollen73S': Pollen73SDataset,
    'Pollen23E': Pollen23EDataset,
    'CretanPollen': CretanPollenDataset,
    'BingolPollen': BingolPollenDataset,
    'CombinedPollen': CombinedPollenDataset
}

PRETRAIN_MODE = False
TRAIN_MODE = True
BATCH_SIZE = 16 if PRETRAIN_MODE else 32
EPOCHS = 400 if PRETRAIN_MODE else 50
LR = 0.1
MOMENTUM = 0.9
SEED = 42
DATASET = 'CombinedPollen' if PRETRAIN_MODE else 'BingolPollen'   # Pollen73S, Pollen23E, CretanPollen or BingolPollen (for transformations)
NUM_CLASSES = num_classes[DATASET]        # 73 for Pollen73S, 23 for Pollen23E, 20 for CretanPollen, 47 for ours
ROOT_DIR = '/home/salih/DGRS_BigEarth/vs_code/Pollen/'
LOG_INTERVAL = 50

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def pretrainUnsupervised(model, train_loader, optimizer, epoch, criterion, wandb):

    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):

        # 32,3,32,32
        # 2 crops of the same image
        images = torch.cat([images[0], images[1]], dim=0)               # comment for one crop case
        images = images.cuda(non_blocking=True)

        # 16,17
        labels = labels.cuda(non_blocking=True)

        # 16
        bsz = labels.shape[0]

        # 32, 128
        features = model(images)                        # model(images)

        # 16, 128 and 16, 128
        #f1, f2 = torch.split(features, [bsz, bsz], dim=0)                   # comment for one crop case

        # 16, 2, 128 -> everything is in order.
        #features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)        # comment for one crop case

        loss = criterion(features)
        if torch.isnan(loss).any():
            print("Loss contains NaN values")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        running_loss += loss.item()
        
        # to avoid the initial zero-th case
        if batch_idx % LOG_INTERVAL == 0 and batch_idx >= LOG_INTERVAL:
            wandb.log({'training loss': running_loss / LOG_INTERVAL})
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(images)//2, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss / LOG_INTERVAL))

def pretrain_model(model, criterion, optimizer, train_loader, wandb):
    model.train()
    running_loss = 0.0

    for epoch in range(EPOCHS):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % LOG_INTERVAL == 0 and batch_idx >= LOG_INTERVAL:
                wandb.log({'training loss': running_loss / LOG_INTERVAL})
                
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(images)//2, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), running_loss / LOG_INTERVAL))
                running_loss = 0.0

def test(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    accuracy = metrics.accuracy(torch.tensor(all_preds), torch.tensor(all_labels))
    # Precision, recall, F1
    precision = metrics.precision(torch.tensor(all_preds), torch.tensor(all_labels), average='macro')
    recall = metrics.recall(torch.tensor(all_preds), torch.tensor(all_labels), average='macro')
    f1_torch = metrics.f1(torch.tensor(all_preds), torch.tensor(all_labels), average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Macro-F1: {f1:.4f}')

    print(f'Accuracy {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 (torch): {f1_torch:.4f}')
    print(f'F1: {f1:.4f}')


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    setSeed(SEED)
    root_dir = ROOT_DIR

    mean_stds = {
        'Pollen73S': ([0.584, 0.522, 0.583], [0.1, 0.14, 0.112]),
        'Pollen23E': ([0.537, 0.554, 0.58], [0.127, 0.14, 0.174]),
        'CretanPollen': ([0.609, 0.431, 0.531], [0.076, 0.158, 0.098]),
        'BingolPollen': ([0.774, 0.702, 0.761], [0.058, 0.162, 0.068]),
        'CombinedPollen': ([0.609, 0.431, 0.531], [0.076, 0.158, 0.098])}

    # SimCLR augmentations
    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
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

    # Create the dataset
    #dataset = PollenDataset(root_dir, transform=train_transform)
    if PRETRAIN_MODE:
        #dataset = PollenDataset(root_dir, transform=train_transform)
        dataset = datasets[DATASET](root_dir, transform=TwoCropTransform(train_transform))  # train_transform
    else:
        dataset = datasets[DATASET](root_dir, transform=val_transform)  # train_transform
    
    # Extract labels for stratified splitting
    labels = [dataset[i][1] for i in range(len(dataset))]

    # Define stratified splitter
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

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

    
    if PRETRAIN_MODE:
        # Pretrain model
        #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)                        # ImageNet pretrained ResNet-50 model
        # Remove the last layer
        #model.fc = torch.nn.Identity()
        #model = model.to(device)

        model = torch.hub.load("nx-ai/vision-lstm", "vil2-base")                                  # ImageNet-1K pretrained Vil2 backbone
        model.head = None
        #model = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)     # Pretrained Mamba backbone
        model = model.to(device)

        criterion = SimCLRLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.0001)

        for epoch in range(1, EPOCHS + 1):
            # adjust the lr
            #lr = LR
            #eta_min = lr * (0.1**3)
            #lr = (
            #    eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / EPOCHS)) / 2
            #)
        
            #for param_group in optimizer.param_groups:
            #    param_group["lr"] = lr
            #print("The learning rate is {}".format(lr))

            pretrainUnsupervised(model, train_loader, optimizer, epoch, criterion, wandb)
            scheduler.step()
        # save the last model
        torch.save(model.state_dict(), f"experiments/combined/pretrain/xlstm_pretrained_{LR}_{BATCH_SIZE}batch_{EPOCHS}epoch.pt")
        print(f"Model saved to experiments/combined/pretrain/xlstm_pretrained_{LR}_{BATCH_SIZE}batch_{EPOCHS}epoch.pt")
    elif TRAIN_MODE:

        # Experiment on different loss functions and lr
        lrs = [0.01, 0.001, 0.0001]
        loss_functions = [torch.nn.CrossEntropyLoss()]      #, FocalLoss(task_type='multi-class', num_classes=NUM_CLASSES)
        best_lr = None
        best_loss_func = None
        best_f1 = 0
        for criterion in loss_functions:
            for lr in lrs:
                print(f"Training with {criterion} and lr={lr}")
                # Create model and optimizer
                #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)                        # pretrained ResNet-50 model
                model = models.resnet50(weights=None)
                # Load the pretrained model
                state_dict = torch.load('experiments/combined/pretrain/resnet50_pretrained_0.1_128batch_400epoch.pt', map_location="cpu")
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # Remove "module." prefix if present
                cudnn.benchmark = True
                model.load_state_dict(new_state_dict, strict=False)

                # Modify the final layer for classification
                model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
                model = model.to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                model_acc = 0
                accs_last10 = np.zeros(10)
                f1s_last10 = np.zeros(10)
                for epoch in range(EPOCHS):
                    model.train()
                    for images, labels in train_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    val_preds = []
                    val_labels = []
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for images, labels in val_loader:
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
                        accs_last10[epoch % 10] = accuracy
                        f1s_last10[epoch % 10] = f1
                        print(f"Epoch {epoch}: Validation accuracy: {accuracy}")
                        print(f"Epoch {epoch}: Validation F1: {f1}")
                
                with open('results.txt', 'a') as f:
                    f.write(f"Loss: {criterion}, lr: {lr}, F1: {np.median(f1s_last10)}, Accuracy: {np.median(accuracy)}\n")

                if np.median(f1s_last10) > best_f1:
                    best_f1 = f1
                    best_loss_fun = criterion
                    best_lr = lr
                    best_model = model

        print(f"Best F1: {best_f1} with loss: {best_loss_fun} and lr: {best_lr}")

        # save the best model
        torch.save(best_model.state_dict(), f"experiments/{DATASET}/resnet50_contrastivepretrained_lr01_128batch_400epoch_{best_loss_fun}_lr{best_lr}_{BATCH_SIZE}batch_{EPOCHS}epoch.pt")
        print(f"Model saved to experiments/{DATASET}/resnet50_contrastivepretrained_lr01_128batch_400epoch_{best_loss_fun}_lr{best_lr}_{BATCH_SIZE}batch_{EPOCHS}epoch.pt")
    else:
        # Load the model from local
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        state_dict = torch.load('experiments/BingolPollen/resnet50_pretrained_CrossEntropyLoss()_lr0.001_32batch_50epoch.pt', map_location="cpu")

        model.load_state_dict(state_dict)

        # TEST_MODE with whatever is the best validation model
        test(model, test_loader)