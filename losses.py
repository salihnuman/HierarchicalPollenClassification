import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    


######### Taken from https://github.com/HobbitLong/SupContrast #########
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    


######### Taken from https://github.com/arashkhoeini/infonce/blob/main/infonce/infonce.py #########
class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Computes the InfoNCE loss.
        
        Args:
            features (torch.Tensor): The feature matrix of shape [2 * batch_size, feature_dim], 
                                     where features[:batch_size] are the representations of 
                                     the first set of augmented images, and features[batch_size:] 
                                     are the representations of the second set.
        
        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        # Normalize features to have unit norm
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Get batch size
        batch_size = features.shape[0] // 2
        
        # Construct labels where each sample's positive pair is in the other view
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarities by setting the diagonal elements to -inf
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class SupervisedInfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07, supervised=False):
        """
        kappa_init: Float, = 1 / temperature
        n_samples: int, how many samples to draw from the vMF distributions
        supervised: bool, whether to define positivity/negativity from class labels (target) or to ignore them
                    and only consider the two crops of the same image as positive
        """
        super().__init__()

        self.temperature = temperature
        self.supervised = supervised

    def forward(self, features, target):

        features = F.normalize(features, dim=-1)

        # Calculate similarities
        sim = features.matmul(features.transpose(-2, -1)) / self.temperature

        # Build positive and negative masks
        mask = (target.unsqueeze(1) == target.t().unsqueeze(0)).float()
        pos_mask = mask - torch.diag(torch.ones(mask.shape[0], device=mask.device))
        neg_mask = 1 - mask
        
        # Things with mask = 0 should be ignored in the sum.
        # If we just gave a zero, it would be log sum exp(0) != 0
        # So we need to give them a small value, with log sum exp(-1000) \approx 0
        pos_mask_add = neg_mask * (-1000)
        neg_mask_add = pos_mask * (-1000)

        # calculate the standard log contrastive loss for each vmf sample ([batch])
        log_infonce_per_example = (sim * pos_mask + pos_mask_add).logsumexp(-1) - (sim * neg_mask + neg_mask_add).logsumexp(-1)

        # Calculate loss ([1])
        log_infonce = torch.mean(log_infonce_per_example)
        return -log_infonce
    

# Inspired from https://github.com/thunderInfy/simclr/blob/master/utils/ntxent.py
class SimCLRLoss(torch.nn.Module):
    def __init__(self, tau=0.1):
        super(SimCLRLoss, self).__init__()
        self.tau = tau

    def forward(self, features):
        batch_size = features.shape[0] // 2
        a = features[:batch_size]
        b = features[batch_size:]
        a_norm = torch.norm(a, dim=1).reshape(-1, 1)
        a_cap = torch.div(a, a_norm)
        b_norm = torch.norm(b, dim=1).reshape(-1, 1)
        b_cap = torch.div(b, b_norm)
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
        b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
        sim_by_tau = torch.div(sim, self.tau)
        exp_sim_by_tau = torch.exp(sim_by_tau)
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
        exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), self.tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        num_by_den = torch.div(numerators, denominators)
        neglog_num_by_den = -torch.log(num_by_den)
        return torch.mean(neglog_num_by_den)



class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, dataset):
        super(HierarchicalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.gene_to_family = {}
        self._get_gene_to_family_mapping()
        self.idx_to_species = {idx: species for species, idx in dataset.species_to_idx.items()}
    
    def _get_gene_to_family_mapping(self):
        with open(os.path.join('gene_to_family.txt'), 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            self.gene_to_family[line.split("\t")[0]] = line.split("\t")[1]

    def getGenus(self, label):
        species = self.idx_to_species[label]
        idx = species.find(" ")
        if idx == -1:
            return species
        return species[:idx]

    def getFamily(self, label):
        genus = self.getGenus(label)
        return self.gene_to_family[genus]

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        # Compute the hierarchical loss
        
        # Get predicted class indices
        pred_labels = torch.argmax(outputs, dim=1)

        # Convert tensors to lists of ints for mapping
        true_labels = targets.tolist()
        pred_labels = pred_labels.tolist()

        # Apply getGenus and getFamily to each label in the batch
        true_genera = [self.getGenus(lbl) for lbl in true_labels]
        pred_genera = [self.getGenus(lbl) for lbl in pred_labels]
        genus_mask = [tg != pg for tg, pg in zip(true_genera, pred_genera)]

        true_families = [self.getFamily(lbl) for lbl in true_labels]
        pred_families = [self.getFamily(lbl) for lbl in pred_labels]
        family_mask = [tf != pf for tf, pf in zip(true_families, pred_families)]

        # Convert masks to tensors
        genus_mask = torch.tensor(genus_mask, dtype=torch.float32, device=outputs.device)
        family_mask = torch.tensor(family_mask, dtype=torch.float32, device=outputs.device)

        weights = torch.ones(batch_size, device=outputs.device) + 0.5 * genus_mask + 0.5 * family_mask
        return (ce_loss * weights).mean()

class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi-task loss (learns log variances)."""
    def __init__(self, num=3):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num, dtype=torch.float32))

    def forward(self, *losses):
        weighted_losses = 0
        for i, loss in enumerate(losses):
            weighted_losses += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return weighted_losses