import torch
import timm
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
#from transformers import AutoModel
#from mambavision import create_model
from torchvision.models.swin_transformer import SwinTransformer

def inspect_model_structure(model):
    """Inspect the structure of your model"""
    print("=== Model Attributes ===")
    for attr in dir(model):
        if not attr.startswith('_'):
            try:
                value = getattr(model, attr)
                if hasattr(value, '__len__') and not isinstance(value, str):
                    print(f"{attr}: {type(value)} (length: {len(value)})")
                elif hasattr(value, 'shape'):
                    print(f"{attr}: {type(value)} (shape: {value.shape})")
                else:
                    print(f"{attr}: {type(value)}")
            except:
                print(f"{attr}: <could not access>")
    
    print("\n=== Model Children ===")
    for name, child in model.named_children():
        print(f"{name}: {type(child)}")
    
    print("\n=== Model Parameters (first 10) ===")
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < 10:
            print(f"{name}: {param.shape}")
        elif i == 10:
            print("... (truncated)")
            break


architectures = ['resnet50', 'xlstm', 'vim', 'swin']
class BackBone(nn.Module):
    def freeze_early_resnet_layers(self, model, num_layers=1):
        # model.backbone.model is the torchvision resnet50
        layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2'][:num_layers+1]
        for name, param in self.model.named_parameters():
            if any(name.startswith(layer) for layer in layers_to_freeze):
                param.requires_grad = False

    def freeze_early_swin_layers(self, num_stages=1, freeze_patch_embed=True):
        """
        Freeze early stages of SWIN transformer from timm
        
        Args:
            num_stages: Number of stages to freeze (1-4 for your model)
            freeze_patch_embed: Whether to freeze patch embedding (default: True)
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = 0
        
        print(f"Freezing first {num_stages} stage(s) of SWIN model...")
        
        # Freeze patch embedding
        if freeze_patch_embed and hasattr(self.model, 'patch_embed'):
            patch_params = 0
            for param in self.model.patch_embed.parameters():
                param.requires_grad = False
                patch_params += param.numel()
                frozen_params += param.numel()
            print(f"✓ Frozen patch_embed ({patch_params:,} parameters)")
        
        # Freeze early layers (stages)
        if hasattr(self.model, 'layers'):
            num_available_layers = len(self.model.layers)
            layers_to_freeze = min(num_stages, num_available_layers)
            
            for i in range(layers_to_freeze):
                layer_params = 0
                for param in self.model.layers[i].parameters():
                    param.requires_grad = False
                    layer_params += param.numel()
                    frozen_params += param.numel()
                print(f"✓ Frozen layers[{i}] (stage {i}) - {layer_params:,} parameters")
            
            if num_stages > num_available_layers:
                print(f"Warning: Requested {num_stages} stages, but model only has {num_available_layers}")
        
        # Summary
        trainable_params = total_params - frozen_params
        print(f"\n=== Freezing Summary ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen percentage: {100 * frozen_params / total_params:.1f}%")
    
    def freeze_early_xlstm_layers(self, num_layers=1):
        # Example: freeze the stem and first num_layers blocks
        # Adjust the attribute names as needed based on the actual model structure
        if hasattr(self.model, 'stem'):
            for param in self.model.stem.parameters():
                param.requires_grad = False
        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(self.model.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False

    def __init__(self, architecture, pretrained=True, device = 'cuda', freeze_early_layers=0):
        super(BackBone, self).__init__()
        
        assert architecture in architectures, f"Invalid architecture: {architecture}. Choose one of {architectures}"

        if architecture == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            self.out_features = self.model.fc.in_features
            self.model.fc = nn.Identity()  # Remove the final classification layer
        elif architecture == "xlstm":
            self.model = torch.hub.load("nx-ai/vision-lstm", "vil2-base")
            self.out_features = self.model.head.in_features
            self.model.head = nn.Identity()
        elif architecture == "vim":
            #self.model = AutoModel.from_pretrained("nvidia/MambaVision-S-1K", trust_remote_code=True)
            self.model = create_model('mamba_vision_T', pretrained=True, model_path="/tmp/mambavision_tiny_1k.pth.tar")
        elif architecture == "swin":
            self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
            self.out_features = self.model.head.in_features
            self.model.reset_classifier(0)
        
        if freeze_early_layers > 0 and architecture == 'resnet50':
            self.freeze_early_resnet_layers(self.model, num_layers=freeze_early_layers)
        if freeze_early_layers > 0 and architecture == 'swin':
            self.freeze_early_swin_layers(num_stages=freeze_early_layers)
        if freeze_early_layers > 0 and architecture == 'xlstm':
            self.freeze_early_xlstm_layers(num_layers=freeze_early_layers)

    def forward(self, x):
        output = self.model(x)
        return output


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, device = "cuda"):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        return self.fc(x)

class HierarchicalPollenModel(nn.Module):
    def __init__(self, backbone_architecture='resnet50', pretrained=True, hierarchy_level = 2, freeze_early_layers=0):
        super(HierarchicalPollenModel, self).__init__()
        self.backbone = BackBone(architecture=backbone_architecture, pretrained=pretrained, freeze_early_layers=freeze_early_layers)
        self.hierarchy_level = hierarchy_level
        if hierarchy_level == 2:
            self.familia_classifier = Classifier(in_features=self.backbone.out_features, out_features=47)
            self.species_classifier = Classifier(in_features=self.backbone.out_features, out_features=274)
        elif hierarchy_level == 3:
            self.familia_classifier = Classifier(in_features=self.backbone.out_features, out_features=47)
            self.genus_classifier = Classifier(in_features=self.backbone.out_features, out_features=145)
            self.species_classifier = Classifier(in_features=self.backbone.out_features, out_features=274)
        else:
            raise ValueError("Invalid hierarchy level. Choose either 2 or 3.")

    def forward(self, x):
        features = self.backbone(x)
        if self.hierarchy_level == 2:
            output_familia = self.familia_classifier(features)
            output_species = self.species_classifier(features)
            output = {
                'familia': output_familia,
                'species': output_species
            }
        elif self.hierarchy_level == 3:
            output_familia = self.familia_classifier(features)
            output_genus = self.genus_classifier(features)
            output_species = self.species_classifier(features)
            output = {
                'familia': output_familia,
                'genus': output_genus,
                'species': output_species
            }
        else:
            raise ValueError("Invalid hierarchy level. Choose either 2 or 3.")
        return output

class PollenModel(nn.Module):
    def __init__(self, backbone_architecture='resnet50', stage = "familia", pretrained=True, freeze_early_layers=0):
        super(PollenModel, self).__init__()
        self.backbone = BackBone(architecture=backbone_architecture, pretrained=pretrained, freeze_early_layers=freeze_early_layers)
        self.classifier = Classifier(in_features=self.backbone.out_features, out_features=47 if stage == "familia" else 274)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output