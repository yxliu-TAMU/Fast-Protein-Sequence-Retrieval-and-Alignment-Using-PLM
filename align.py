import argparse
import random
import itertools
import os
import shutil
import tempfile

import numpy as np
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig


def set_seed(seed):
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MsaPairAlignmentDataset(IterableDataset):
    def __init__(self, tokenizer, file_path, offsets=None):
        assert os.path.isdir(file_path), "Expected a directory of .txt files"
        print('Loading the protein alignment dataset...')
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.offsets = offsets

    def __iter__(self):
        if self.offsets is not None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id
            offset_start = self.offsets[worker_id]
            offset_end = self.offsets[worker_id + 1] if worker_id + 1 < len(self.offsets) else None
        else:
            offset_start = 0
            offset_end = None
            worker_id = 0

        for file_name in sorted(os.listdir(self.file_path)):
            if not file_name.endswith(".txt"): continue
            file_path = os.path.join(self.file_path, file_name)
            with open(file_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                ref_line = next((l for l in lines if l.startswith("Reference:")), None)
                aln_line = next((l for l in lines if l.startswith("Aligned:")), None)
                if ref_line is None or aln_line is None:
                    continue

                aligned_src = ref_line.replace("Reference:", "").strip()
                aligned_tgt = aln_line.replace("Aligned:", "").strip()

                # Extract unaligned sequences by removing gaps
                src = aligned_src.replace('-', '')
                tgt = aligned_tgt.replace('-', '')
                if not src or not tgt:
                    continue

                token_src = self.tokenizer.tokenize(src)
                token_tgt = self.tokenizer.tokenize(tgt)
                wid_src = self.tokenizer.convert_tokens_to_ids(token_src)
                wid_tgt = self.tokenizer.convert_tokens_to_ids(token_tgt)
                ids_src = self.tokenizer.prepare_for_model(wid_src, return_tensors='pt')['input_ids']
                ids_tgt = self.tokenizer.prepare_for_model(wid_tgt, return_tensors='pt')['input_ids']
                if ids_src.dim() == 1:
                    ids_src = ids_src.unsqueeze(0)
                if ids_tgt.dim() == 1:
                    ids_tgt = ids_tgt.unsqueeze(0)

                if len(ids_src[0]) <= 2 or len(ids_tgt[0]) <= 2:
                    continue

                bpe2word_map_src = list(range(len(token_src)))
                bpe2word_map_tgt = list(range(len(token_tgt)))

                yield (worker_id, ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt, src, tgt,aligned_src,aligned_tgt)

def find_offsets(filename, num_workers):
    if num_workers <= 1:
        return None
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_workers
        offsets = [0]
        for i in range(1, num_workers):
            f.seek(chunk_size * i)
            while True:
                try:
                    f.readline()
                    break
                except UnicodeDecodeError:
                    f.seek(f.tell() - 1)
            offsets.append(f.tell())
    return offsets


def word_align(args, model, tokenizer):
    def collate(examples):
        worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt,gts_src, gts_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt,gts_src, gts_tgt

    offsets = None  # Disabled because args.data_file is a directory
    dataset = MsaPairAlignmentDataset(tokenizer, file_path=args.data_file, offsets=offsets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers)

    model.to(args.device)
    model.eval()
    iterator = trange(0, desc="Aligning")
    all_metrics = []
    with open(args.output_file, 'w') as writer:
        for batch in dataloader:
            with torch.no_grad():
                worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt, gts_src, gts_tgt = batch
                # Forward pass
                outputs_src = model(input_ids=ids_src.to(args.device), output_hidden_states=True)
                outputs_tgt = model(input_ids=ids_tgt.to(args.device), output_hidden_states=True)
                hidden_src = outputs_src.hidden_states[args.align_layer][:,1:-1,:]  # [B, L, D]
                hidden_tgt = outputs_tgt.hidden_states[args.align_layer][:,1:-1,:]
                #print(ids_src.shape,hidden_src.shape,len(sents_src[0]))
                # Compute dot-product attention
                if args.extraction == "softmax":
                    sim_matrix = torch.einsum("bid,bjd->bij", hidden_src, hidden_tgt)
                    probs = torch.nn.functional.softmax(sim_matrix, dim=-1)
                elif args.extraction == "ot":
                    hidden_src_norm = torch.nn.functional.normalize(hidden_src, dim=-1)
                    hidden_tgt_norm = torch.nn.functional.normalize(hidden_tgt, dim=-1)
                    cost_matrix = 1 - torch.einsum("bid,bjd->bij", hidden_src_norm, hidden_tgt_norm)
                    eps = 1e-3
                    probs = []
                    for b in range(cost_matrix.size(0)):
                        M = cost_matrix[b]
                        u = torch.ones(M.size(0), device=M.device)
                        v = torch.ones(M.size(1), device=M.device)
                        K = torch.exp(-M / eps)
                        for _ in range(10):
                            u = 1.0 / (K @ v)
                            v = 1.0 / (K.T @ u)
                        probs_b = torch.diag(u) @ K @ torch.diag(v)
                        probs.append(probs_b.unsqueeze(0))
                    probs = torch.cat(probs, dim=0)
                else:
                    raise ValueError("Invalid extraction method: choose 'softmax' or 'ot'")

                for i, (src, tgt, gsrc, gtgt) in enumerate(zip(sents_src, sents_tgt, gts_src, gts_tgt)):
                    alignment = []
                    for src_idx, prob in enumerate(probs[i]):
                        tgt_idx = torch.argmax(prob).item()
                        alignment.append(f"{src_idx}-{tgt_idx}")

                    alignment_str = ' '.join(alignment)
                    writer.write(alignment_str + ' ')

                    # Evaluate using preloaded ground truth
                    #print(alignment_str)
                    #print(gsrc,gtgt)
                    p, r, f1,aer = evaluate_alignment(alignment_str, gsrc, gtgt)
                    all_metrics.append((p, r, f1,aer))
                    #print(p,r,f1,aer)
            iterator.update(len(ids_src))
    print(summarize_metrics(all_metrics))


def evaluate_alignment(pred_alignments, src_seq, tgt_seq):
    """
    Compare predicted alignments to ground truth based on position of non-gap characters.
    Assumes src_seq and tgt_seq are aligned and of equal length.
    Returns precision, recall, F1, and AER.
    """
    # Ground truth: align positions with both non-gap characters
    ground_truth = [(i, i) for i, (a, b) in enumerate(zip(src_seq, tgt_seq)) if a != '-' and b != '-']
    ground_truth_set = set(ground_truth)

    # Predicted alignments
    pred_pairs = set(tuple(map(int, p.split('-'))) for p in pred_alignments.split())

    true_positives = len(pred_pairs & ground_truth_set)
    pred_total = len(pred_pairs)
    gold_total = len(ground_truth_set)

    precision = true_positives / pred_total if pred_total else 0
    recall = true_positives / gold_total if gold_total else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    aer = 1.0 - (2 * true_positives) / (pred_total + gold_total) if (pred_total + gold_total) > 0 else 1.0

    return precision, recall, f1, aer

def summarize_metrics(metrics):
    if not metrics:
        return 0.0, 0.0, 0.0
    precisions, recalls, f1s,aer = zip(*metrics)
    return sum(precisions)/len(metrics), sum(recalls)/len(metrics), sum(f1s)/len(metrics),sum(aer)/len(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", default="softmax", choices=["softmax", "ot"], help="Alignment extraction method: softmax or ot")
    parser.add_argument("--data_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--model_name_or_path", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--align_layer", default=5, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    word_align(args, model, tokenizer)


if __name__ == "__main__":
    main()
