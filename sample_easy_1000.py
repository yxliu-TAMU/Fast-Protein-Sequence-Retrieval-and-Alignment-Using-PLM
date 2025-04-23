import argparse
import random

def read_pairs(input_file):
    pairs = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        ref_line = lines[i].strip().replace("Reference: ", "")
        aln_line = lines[i+1].strip().replace("Aligned:   ", "")
        pairs.append((ref_line, aln_line))
    return pairs

def write_pairs(pairs, output_file):
    with open(output_file, 'w') as f:
        for ref, aln in pairs:
            f.write(f"Reference: {ref}\n")
            f.write(f"Aligned:   {aln}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the full test_easy.txt")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the sampled 1000 pairs")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    pairs = read_pairs(args.input_file)
    print(f"Total available pairs in test_easy: {len(pairs)}")

    if len(pairs) < args.num_samples:
        raise ValueError(f"Not enough pairs to sample {args.num_samples} sequences!")

    sampled_pairs = random.sample(pairs, args.num_samples)
    write_pairs(sampled_pairs, args.output_file)
    print(f"Saved {args.num_samples} randomly sampled pairs to {args.output_file}")

if __name__ == "__main__":
    main()
