import os
import random
import argparse
import json
from multiprocessing import Pool
from tqdm import tqdm

def parse_fasta(a3m_string):
    sequences = []
    for line in a3m_string.strip().splitlines():
        if line.startswith(">"):
            sequences.append("")
        else:
            sequences[-1] += line.strip()
    return sequences

def process_folder(folder_path):
    a3m_dir = os.path.join(folder_path, "a3m")
    if not os.path.isdir(a3m_dir):
        return None, None, None  # Skip if no a3m subfolder

    a3m_files = [f for f in os.listdir(a3m_dir) if f.endswith(".a3m")]
    if not a3m_files:
        return None, None, None  # No a3m file found
    a3m_path = os.path.join(a3m_dir, a3m_files[0])

    try:
        with open(a3m_path, 'r') as f:
            a3m_str = f.read()
        sequences = parse_fasta(a3m_str)
        if len(sequences) < 2:
            return None, None, None  # Not enough sequences

        ref_seq = sequences[0]
        aligned_sequences = sequences[1:]
        if len(aligned_sequences) < 2:
            return None, None, None  # Not enough for both train and easy

        train_idx, easy_idx = random.sample(range(len(aligned_sequences)), 2)
        train_seq = aligned_sequences[train_idx]
        easy_seq = aligned_sequences[easy_idx]

        def align_pair(aligned):
            ref_aln, aln_aln = [], []
            for r, a in zip(ref_seq, aligned):
                if a == '-':
                    ref_aln.append(r)
                    aln_aln.append('-')
                elif a.islower():
                    ref_aln.append('-')
                    aln_aln.append(a.upper())
                else:
                    ref_aln.append(r)
                    aln_aln.append(a)
            return ("".join(ref_aln), "".join(aln_aln))

        return align_pair(train_seq), align_pair(easy_seq), None
    except Exception as e:
        print(f"[WARN] Skipped folder {folder_path}: {e}")
        return None, None, None

def process_hard_folder(folder_path):
    a3m_dir = os.path.join(folder_path, "a3m")
    if not os.path.isdir(a3m_dir):
        return None

    a3m_files = [f for f in os.listdir(a3m_dir) if f.endswith(".a3m")]
    if not a3m_files:
        return None
    a3m_path = os.path.join(a3m_dir, a3m_files[0])

    try:
        with open(a3m_path, 'r') as f:
            a3m_str = f.read()
        sequences = parse_fasta(a3m_str)
        if len(sequences) < 2:
            return None

        ref_seq = sequences[0]
        aligned_sequences = sequences[1:]
        hard_idx = random.randint(0, len(aligned_sequences)-1)
        hard_seq = aligned_sequences[hard_idx]

        ref_aln, aln_aln = [], []
        for r, a in zip(ref_seq, hard_seq):
            if a == '-':
                ref_aln.append(r)
                aln_aln.append('-')
            elif a.islower():
                ref_aln.append('-')
                aln_aln.append(a.upper())
            else:
                ref_aln.append(r)
                aln_aln.append(a)
        return ("".join(ref_aln), "".join(aln_aln))
    except Exception as e:
        print(f"[WARN] Skipped test folder {folder_path}: {e}")
        return None

def write_pairs(pairs, output_path):
    with open(output_path, 'w') as f:
        for ref, aln in pairs:
            f.write(f"Reference: {ref}\n")
            f.write(f"Aligned:   {aln}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3m_dir", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--test_easy_output", type=str, required=True)
    parser.add_argument("--test_hard_output", type=str, required=True)
    parser.add_argument("--save_split", type=str, required=True, help="JSON file to save folder split info")
    parser.add_argument("--num_test_folders", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_folders = sorted([os.path.join(args.a3m_dir, d) for d in os.listdir(args.a3m_dir) if os.path.isdir(os.path.join(args.a3m_dir, d))])
    print(f"Total folders found: {len(all_folders)}")

    test_hard_folders = set(random.sample(all_folders, args.num_test_folders))
    train_easy_folders = [f for f in all_folders if f not in test_hard_folders]

    with open(args.save_split, 'w') as f:
        json.dump({"test_hard": list(test_hard_folders), "train_easy": train_easy_folders}, f, indent=2)

    print(f"Preparing train and test_easy from {len(train_easy_folders)} folders...")
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_folder, train_easy_folders), total=len(train_easy_folders)))

    train_pairs = [res[0] for res in results if res[0] is not None]
    test_easy_pairs = [res[1] for res in results if res[1] is not None]

    write_pairs(train_pairs, args.train_output)
    write_pairs(test_easy_pairs, args.test_easy_output)
    print(f"Saved {len(train_pairs)} training pairs and {len(test_easy_pairs)} test easy pairs.")

    print(f"Preparing test_hard from {len(test_hard_folders)} folders...")
    with Pool(processes=args.num_workers) as pool:
        hard_results = list(tqdm(pool.imap_unordered(process_hard_folder, test_hard_folders), total=len(test_hard_folders)))
    test_hard_pairs = [res for res in hard_results if res is not None]

    write_pairs(test_hard_pairs, args.test_hard_output)
    print(f"Saved {len(test_hard_pairs)} test hard pairs.")

if __name__ == "__main__":
    main()
