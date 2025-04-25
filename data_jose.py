import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple
import pdb
from transformers import EsmTokenizer, EsmModel

class ProteinAlignmentDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer:EsmTokenizer, model: EsmModel, single_file: bool = False):
        self.tokenizer = tokenizer
        self.model = model
        if single_file:
            self.data = self._load_data_txt(data_dir)
        else:
            self.data = self._load_data(data_dir)

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
                
                tokenized = self.tokenizer(ref, return_tensors="pt")
                output = self.model(tokenized.input_ids, tokenized.attention_mask)
                ref_ids = output.last_hidden_state.squeeze()

                tokenized = self.tokenizer(hyp, return_tensors="pt")
                output = self.model(tokenized.input_ids, tokenized.attention_mask)
                hyp_ids = output.last_hidden_state.squeeze()

                # # Filter sequences containing "<unk>"
                # if self.token2idx["<unk>"] in ref_ids or self.token2idx["<unk>"] in hyp_ids:
                #     continue

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

                data.append((ref_ids, hyp_ids, insert_counts, delete_labels, ref_raw, ali_raw))
        return data

    def _load_data_txt(self, file_path: str) -> List[Tuple[List[int], List[int], List[float], List[int], str, str]]:
        data = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                ref_raw = lines[i].strip().split("Reference:")[-1].strip()
                ali_raw = lines[i + 1].strip().split("Aligned:")[-1].strip()
                ref = ref_raw.replace('-', '')
                hyp = ali_raw.replace('-', '')

                tokenized = self.tokenizer(ref, return_tensors="pt")
                output = self.model(tokenized.input_ids, tokenized.attention_mask)
                ref_ids = output.last_hidden_state.squeeze()

                tokenized = self.tokenizer(hyp, return_tensors="pt")
                output = self.model(tokenized.input_ids, tokenized.attention_mask)
                hyp_ids = output.last_hidden_state.squeeze()

                # # Filter sequences containing "<unk>"
                # if self.token2idx["<unk>"] in ref_ids or self.token2idx["<unk>"] in hyp_ids:
                #     continue

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

def collate_fn(batch):
    ref_seqs, hyp_seqs, insert_targets, delete_targets, ref_raws, ali_raws = zip(*batch)

    ref_pad = torch.nn.utils.rnn.pad_sequence(ref_seqs, padding_value=0)
    hyp_pad = torch.nn.utils.rnn.pad_sequence(hyp_seqs, padding_value=0)
    insert_pad = torch.nn.utils.rnn.pad_sequence(insert_targets, padding_value=0)
    delete_pad = torch.nn.utils.rnn.pad_sequence(delete_targets, padding_value=0)

    ref_mask = (ref_pad == 0).transpose(0, 1)
    hyp_mask = (hyp_pad == 0).transpose(0, 1)

    return ref_pad, hyp_pad, insert_pad, delete_pad, ref_mask, hyp_mask, ref_raws, ali_raws


def get_dataloaders(data_dir: str, batch_size: int = 16, single_file: bool = False, seed: int = 42, train: bool = True):
    esm_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    esm_model = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D')
    vocab = esm_tokenizer.get_vocab().keys()
    dataset = ProteinAlignmentDataset(data_dir, esm_tokenizer, esm_model, single_file=single_file)

    if train:
        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader, vocab
    else:
        full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return full_loader, vocab

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
    train_loader, val_loader, vocab = get_dataloaders("database/toy_dataset/", batch_size=4, single_file=False)

    """
    Reconstruction tests do not work because ref_ids and hyp_ids are no longer embeddings of the tokens in the sequence
    but rather an encoded sequence of embeddings. I am not sure how to test if my method is sound. Maybe by checking the
    shape of the ref_ids or by making sure a known raw input matches outputs from the dataloader and directly from the ESM model
    """


    # # Test reconstruction on the validation set
    # val_dataset = val_loader.dataset  # Subset object
    # print(f"Validation set size (folder): {len(val_dataset)}")

    # # Loop over first 5 examples for reconstruction
    # for idx in range(min(5, len(val_dataset))):
    #     ref_ids, hyp_ids, insert_counts, delete_labels, ref_raw, ali_raw = val_dataset[idx]
    #     ref_seq = ''.join([vocab[i] for i in ref_ids.tolist()])
    #     hyp_seq = ''.join([vocab[i] for i in hyp_ids.tolist()])[5:]
        
    #     recon_ref, recon_ali = reconstruct_alignment(ref_seq, hyp_seq, insert_counts.tolist(), delete_labels.tolist())
    #     assert recon_ref == ref_raw, f"Reconstructed reference does not match original: {recon_ref} != {ref_raw}"
    #     assert recon_ali == ali_raw, f"Reconstructed aligned does not match original: {recon_ali} != {ali_raw}"

    # # ------------------------------
    # # Example 2: Single File Dataset with Random Sampling
    # # ------------------------------
    # print("\n=== Single File Dataset Random Sampling Reconstruction ===")
    # single_dataset = ProteinAlignmentDataset("database/large_dataset/train.txt", vocab, single_file=True)
    # print(f"Total samples in single file dataset: {len(single_dataset)}")

    # # Randomly select 100 examples
    # sample_indices = torch.randperm(len(single_dataset))[:100]

    # for count, idx in enumerate(sample_indices.tolist()):
    #     ref_ids, hyp_ids, insert_counts, delete_labels, ref_raw, ali_raw = single_dataset[idx]
    #     ref_seq = ''.join([vocab[i] for i in ref_ids.tolist()])
    #     hyp_seq = ''.join([vocab[i] for i in hyp_ids.tolist()])[5:]
        
    #     recon_ref, recon_ali = reconstruct_alignment(ref_seq, hyp_seq, insert_counts.tolist(), delete_labels.tolist())
    #     assert recon_ref == ref_raw, f"Reconstructed reference does not match original: {recon_ref} != {ref_raw}"
    #     assert recon_ali == ali_raw, f"Reconstructed aligned does not match original: {recon_ali} != {ali_raw}"
