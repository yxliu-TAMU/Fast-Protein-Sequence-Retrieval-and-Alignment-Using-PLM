import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import pdb

class ProteinAlignmentDataset(Dataset):
    def __init__(self, data_dir: str, vocab: str):
        self.vocab = vocab
        self.token2idx = {aa: i for i, aa in enumerate(vocab)}
        self.idx2token = {i: aa for aa, i in self.token2idx.items()}
        self.pad_idx = self.token2idx['<PAD>']
        self.bos_idx = self.token2idx['<BOS>']
        self.data = self._load_data(data_dir)
    """
    def _load_data(self, folder: str) -> List[Tuple[List[int], List[int], List[float], List[int], str, str]]:
        data = []
        for fname in os.listdir(folder):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(folder, fname), "r") as f:
                lines = f.readlines()
                #pdb.set_trace()
                ref_raw = lines[0].strip().split("Reference:")[-1].strip()
                ali_raw = lines[1].strip().split("Aligned:")[-1].strip()
                ref = ref_raw.replace('-', '')
                hyp = ali_raw.replace('-', '')

                #hyp = '<BOS>' + hyp
                insert_counts = [0] * (len(hyp)+1)
                delete_labels = [1]  # BOS token is always marked as deleted

                ref_ptr = 0
                ali_ptr = 0

                while ali_ptr < len(ali_raw):
                    if ali_raw[ali_ptr] == '-':
                        insert_counts[len(delete_labels) - 1] += 1
                        ali_ptr += 1
                    else:
                        if ref_raw[ali_ptr] == '-':
                            delete_labels.append(1)
                        else:
                            delete_labels.append(0)
                            ref_ptr += 1
                        ali_ptr += 1

                ref_ids = [self.token2idx.get(c, self.token2idx['<UNK>']) for c in ref]
                hyp_ids = [self.bos_idx] + [self.token2idx.get(c, self.token2idx['<UNK>']) for c in hyp]
                #pdb.set_trace()
                data.append((ref_ids, hyp_ids, insert_counts, delete_labels, ref_raw, ali_raw))
        return data
    """

    def _load_data(self, folder: str) -> List[Tuple[List[int], List[int], List[float], List[int], str, str]]:
        data = []
        for fname in os.listdir(folder):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(folder, fname), "r") as f:
                lines = f.readlines()
                ref_raw = lines[0].strip().split("Reference:")[-1].strip()
                ali_raw = lines[1].strip().split("Aligned:")[-1].strip()
                ref = ref_raw.replace('-', '')
                hyp = ali_raw.replace('-', '')

                ref_ids = [self.token2idx.get(c, self.token2idx['<UNK>']) for c in ref]
                hyp_ids = [self.token2idx.get(c, self.token2idx['<UNK>']) for c in hyp]

                # Filter sequences containing '<UNK>'
                if self.token2idx['<UNK>'] in ref_ids or self.token2idx['<UNK>'] in hyp_ids:
                    continue

                insert_counts = [0] * (len(hyp) + 1)
                delete_labels = [1]  # BOS token is always marked as deleted

                ref_ptr = 0
                ali_ptr = 0

                while ali_ptr < len(ali_raw):
                    if ali_raw[ali_ptr] == '-':
                        insert_counts[len(delete_labels) - 1] += 1
                        ali_ptr += 1
                    else:
                        if ref_raw[ali_ptr] == '-':
                            delete_labels.append(1)
                        else:
                            delete_labels.append(0)
                            ref_ptr += 1
                        ali_ptr += 1

                hyp_ids = [self.bos_idx] + hyp_ids
                data.append((ref_ids, hyp_ids, insert_counts, delete_labels, ref_raw, ali_raw))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ref_ids, hyp_ids, insert_counts, delete_labels, ref_raw, ali_raw = self.data[idx]
        return torch.tensor(ref_ids), torch.tensor(hyp_ids), torch.tensor(insert_counts), torch.tensor(delete_labels), ref_raw, ali_raw

def collate_fn(batch):
    ref_seqs, hyp_seqs, insert_targets, delete_targets, ref_raws, ali_raws = zip(*batch)

    ref_pad = torch.nn.utils.rnn.pad_sequence(ref_seqs, padding_value=0)
    hyp_pad = torch.nn.utils.rnn.pad_sequence(hyp_seqs, padding_value=0)
    insert_pad = torch.nn.utils.rnn.pad_sequence(insert_targets, padding_value=0)
    delete_pad = torch.nn.utils.rnn.pad_sequence(delete_targets, padding_value=0)

    ref_mask = (ref_pad == 0).transpose(0, 1)
    hyp_mask = (hyp_pad == 0).transpose(0, 1)

    return ref_pad, hyp_pad, insert_pad, delete_pad, ref_mask, hyp_mask, ref_raws, ali_raws


def get_dataloader(data_dir: str, batch_size: int = 16):
    vocab = list("ACDEFGHIKLMNPQRSTVWY") + ['<PAD>', '<UNK>', '<BOS>']
    dataset = ProteinAlignmentDataset(data_dir, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return loader, vocab

"""
def reconstruct_alignment(ref: str, hyp: str, insert_counts: List[int], delete_labels: List[int]) -> Tuple[str, str]:
    ref_out, ali_out = [], []
    r_idx = 0
    #pdb.set_trace()
    hyp = hyp[5:]  # Remove the <BOS> token
    delete_labels = delete_labels[1:]  # Remove the <BOS> token
    #reconstruct the aligned sequence
    for i, c in enumerate(hyp):
        if insert_counts[i]>0:
            ali_out.append('-' * insert_counts[i])
        ali_out.append(c)
    ali_out.append('-' * insert_counts[-1])

    #reconstruct the reference sequence
    for i, c in enumerate(hyp):
        #pdb.set_trace()
        if delete_labels[i]==0 and insert_counts[i]==0:
            ref_out.append(ref[r_idx])
            r_idx += 1
        elif delete_labels[i]==0 and insert_counts[i]>0:
            for _ in range(insert_counts[i]+1):
                ref_out.append(ref[r_idx])
                r_idx += 1
        elif delete_labels[i]==1 and insert_counts[i]>0:
            for _ in range(insert_counts[i]):
                ref_out.append(ref[r_idx])
                r_idx += 1
        else:
            ref_out.append('-')
    if insert_counts[-1]>0:
        for _ in range(insert_counts[-1]):
            ref_out.append(ref[r_idx])
            r_idx += 1
    return ''.join(ref_out), ''.join(ali_out)
"""

def reconstruct_alignment(ref: str, hyp: str, insert_counts: List[int], delete_labels: List[int]) -> Tuple[str, str]:
    ref_out, ali_out = [], []
    r_idx = 0
    h_idx = 0  # hypothesis pointer

    # Remove the <BOS> token label from delete_labels (already handled by insert_counts)
    delete_labels = delete_labels[1:]

    # insert_counts length is always len(hyp) + 1
    for i in range(len(hyp)):
        # Insert gaps before current hypothesis token if any
        if insert_counts[i] > 0:
            ali_out.extend(['-'] * insert_counts[i])
            ref_out.extend(ref[r_idx: r_idx + insert_counts[i]])
            r_idx += insert_counts[i]

        # Now handle current token based on deletion label
        if delete_labels[i] == 0:
            # Match case (no deletion): append both tokens
            ali_out.append(hyp[h_idx])
            ref_out.append(ref[r_idx])
            r_idx += 1
        else:
            # Deletion case: hypothesis has token, reference has gap
            ali_out.append(hyp[h_idx])
            ref_out.append('-')

        h_idx += 1

    # Handle insertions after the last hypothesis token
    if insert_counts[-1] > 0:
        ali_out.extend(['-'] * insert_counts[-1])
        ref_out.extend(ref[r_idx: r_idx + insert_counts[-1]])
        r_idx += insert_counts[-1]

    return ''.join(ref_out), ''.join(ali_out)


if __name__ == "__main__":
    loader, vocab = get_dataloader("database/toy_dataset", batch_size=1)
    for ref_ids, hyp_ids, insert_counts, delete_labels, _, _, ref_raws, ali_raws in loader:
        ref_seq = ''.join([vocab[i] for i in ref_ids[:, 0].tolist()])
        hyp_seq = ''.join([vocab[i] for i in hyp_ids[:, 0].tolist()])[5:]  # Exclude "<BOS>"

        insert_counts = insert_counts[:, 0].tolist()
        delete_labels = delete_labels[:, 0].tolist()
        # length of the inputs not matching, and pdb.set_trace()
        if len(insert_counts) != len(hyp_seq) + 1 or len(delete_labels) != len(hyp_seq)+1:
            print(f"Length mismatch: {len(ref_seq)} != {len(ref_raws[0])} or {len(hyp_seq)} != {len(ali_raws[0])}")
            pdb.set_trace()
        recon_ref, recon_ali = reconstruct_alignment(ref_seq, hyp_seq, insert_counts, delete_labels)
        assert recon_ali == ali_raws[0], f"Mismatch in Aligned: {recon_ali} != {ali_raws[0]}"
        assert recon_ref == ref_raws[0], f"Mismatch in Reference: {recon_ref} != {ref_raws[0]}"

        print("Reconstructed alignment matches original for one example.")
    print(len(loader.dataset))
