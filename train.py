import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import ProteinLevenshteinTransformer

#Evaluation Function
def evaluate_insertion_deletion(insert_logits, insert_target, delete_logits, delete_target):
    """
    Compute insertion and deletion evaluation metrics.
    
    Args:
        insert_logits: predicted insert counts (raw floats), shape [T, B]
        insert_target: true insert counts, shape [T, B]
        delete_logits: predicted deletion logits, shape [T, B]
        delete_target: true deletion labels (0 or 1), shape [T, B]

    Returns:
        dict: metrics { 'insert_mae', 'delete_accuracy', 
                        'insert_mae_nonzero', 'insert_detection_rate',
                        'delete_precision', 'delete_recall', 'delete_f1' }
    """
    # --- Insertion Metrics ---
    insert_error = torch.abs(insert_logits - insert_target)
    insert_mae = insert_error.mean().item()  # Original MAE

    nonzero_mask = (insert_target > 0)
    if nonzero_mask.sum() > 0:
        insert_mae_nonzero = insert_error[nonzero_mask].mean().item()
    else:
        insert_mae_nonzero = 0.0  # Handle rare case where no insertions > 0 in batch

    # Detection rate: did the model predict >0 insert where it should?
    insert_detect_pred = (insert_logits > 0.5).float()
    correct_insertion_detect = ((insert_detect_pred == 1) & (insert_target > 0)).sum().item()
    total_insertion_sites = (insert_target > 0).sum().item()
    insertion_detection_rate = correct_insertion_detect / (total_insertion_sites + 1e-8)

    # --- Deletion Metrics ---
    delete_pred = (torch.sigmoid(delete_logits) > 0.5).float()
    delete_accuracy = (delete_pred == delete_target).float().mean().item()  # Original accuracy

    true_positives = ((delete_pred == 1) & (delete_target == 1)).sum().item()
    false_positives = ((delete_pred == 1) & (delete_target == 0)).sum().item()
    false_negatives = ((delete_pred == 0) & (delete_target == 1)).sum().item()

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'insert_mae': insert_mae,
        'delete_accuracy': delete_accuracy,
        'insert_mae_nonzero': insert_mae_nonzero,
        'insert_detection_rate': insertion_detection_rate,
        'delete_precision': precision,
        'delete_recall': recall,
        'delete_f1': f1
    }

# Configuration
DATA_DIR = "database/large_dataset/train.txt"
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LOG1P_INSERT = False
USE_FOCAL_LOSS_DELETE = False
POS_WEIGHT_DELETE = 5.0


# Load data
train_loader, val_loader,vocab = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, single_file=True)
print(f"Train Dataloader: {len(train_loader)} batches")
print(f"Validation Dataloader: {len(val_loader)} batches")
vocab_size = len(vocab)

# Model setup
model = ProteinLevenshteinTransformer(vocab_size=vocab_size).to(DEVICE)

if USE_LOG1P_INSERT:
    def log1p_loss(pred, target):
        return nn.MSELoss()(torch.log1p(torch.clamp(pred, min=0)), torch.log1p(torch.clamp(target, min=0)))
    criterion_insert = log1p_loss
else:
    criterion_insert = nn.MSELoss()

# Delete loss setup
if USE_FOCAL_LOSS_DELETE:
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=0.25):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            ce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_factor = (1 - p_t) ** self.gamma
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            return (alpha_factor * focal_factor * ce_loss).mean()
    criterion_delete = FocalLoss()
else:
    criterion_delete = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT_DELETE]).to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=1e-4)





# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_insert_loss, total_delete_loss = 0.0, 0.0

    for step,(ref, hyp, insert_target, delete_target, ref_mask, hyp_mask,_,_) in enumerate(train_loader):
        ref = ref.to(DEVICE)
        hyp = hyp.to(DEVICE)
        insert_target = insert_target.to(DEVICE).float()
        delete_target = delete_target.to(DEVICE).float()
        ref_mask = ref_mask.to(DEVICE)
        hyp_mask = hyp_mask.to(DEVICE)

        optimizer.zero_grad()
        insert_logits, delete_logits = model(
            ref, hyp,
            src_key_padding_mask=ref_mask,
            tgt_key_padding_mask=hyp_mask,
            memory_key_padding_mask=ref_mask
        )

        insert_loss = criterion_insert(insert_logits, insert_target) 
        delete_loss = criterion_delete(delete_logits.view(-1), delete_target.view(-1))

        loss = insert_loss + delete_loss
        loss.backward()
        optimizer.step()

        total_insert_loss += insert_loss.item()
        total_delete_loss += delete_loss.item()
        if step % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {step} | Insert Loss: {insert_loss.item():.4f} | Delete Loss: {delete_loss.item():.4f}")

    print(f"Epoch {epoch+1}/{EPOCHS} | Insert Loss: {total_insert_loss:.4f} | Delete Loss: {total_delete_loss:.4f}")
    model.eval()
    with torch.no_grad():
        all_metrics = {'insert_mae': 0, 'delete_accuracy': 0, 'insert_mae_nonzero': 0,
                       'insert_detection_rate': 0, 'delete_precision': 0, 'delete_recall': 0, 'delete_f1': 0}
        count = 0

        for ref, hyp, insert_target, delete_target, ref_mask, hyp_mask, _, _ in val_loader:
            ref, hyp = ref.to(DEVICE), hyp.to(DEVICE)
            insert_target, delete_target = insert_target.to(DEVICE).float(), delete_target.to(DEVICE).float()
            ref_mask, hyp_mask = ref_mask.to(DEVICE), hyp_mask.to(DEVICE)

            insert_logits, delete_logits = model(
                ref, hyp,
                src_key_padding_mask=ref_mask,
                tgt_key_padding_mask=hyp_mask,
                memory_key_padding_mask=ref_mask
            )

            metrics = evaluate_insertion_deletion(insert_logits, insert_target, delete_logits, delete_target)
            for k in all_metrics:
                all_metrics[k] += metrics[k]
            count += 1

        avg_metrics = {k: v / count for k, v in all_metrics.items()}

    print(f"Val Insert MAE (all): {avg_metrics['insert_mae']:.4f} | MAE (nonzero): {avg_metrics['insert_mae_nonzero']:.4f} | "
          f"Detection Rate: {avg_metrics['insert_detection_rate']:.4f}")
    print(f"Val Delete Accuracy: {avg_metrics['delete_accuracy']:.4f} | Precision: {avg_metrics['delete_precision']:.4f} | "
          f"Recall: {avg_metrics['delete_recall']:.4f} | F1: {avg_metrics['delete_f1']:.4f}")