import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloader
from model import ProteinLevenshteinTransformer

# Configuration
DATA_DIR = "database/toy_dataset/"
BATCH_SIZE = 16
EPOCHS = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, vocab = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
vocab_size = len(vocab)

# Model setup
model = ProteinLevenshteinTransformer(vocab_size=vocab_size).to(DEVICE)
criterion_insert = nn.MSELoss()
criterion_delete = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_insert_loss, total_delete_loss = 0.0, 0.0

    for ref, hyp, insert_target, delete_target, ref_mask, hyp_mask,_,_ in train_loader:
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

    print(f"Epoch {epoch+1}/{EPOCHS} | Insert Loss: {total_insert_loss:.4f} | Delete Loss: {total_delete_loss:.4f}")
