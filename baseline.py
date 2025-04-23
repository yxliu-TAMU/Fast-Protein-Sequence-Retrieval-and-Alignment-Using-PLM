import os
import tempfile
import torch
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Blast import NCBIXML
from Bio import SeqIO
from data import get_dataloaders
from train import evaluate_insertion_deletion  # Adjust to your actual import path

DATA_DIR = "database/toy_dataset/"
BATCH_SIZE = 1  # One pair at a time
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, vocab = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, single_file=False, train=False)


def write_fasta(seq, file_path, seq_id="seq"):
    with open(file_path, "w") as f:
        f.write(f">{seq_id}\n{seq}\n")


def parse_blast_xml(xml_file):
    """Parse BLAST XML output to extract aligned sequences."""
    with open(xml_file) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        for blast_record in blast_records:
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps:
                    return hsp.sbjct, hsp.query  # aligned_ref, aligned_ali
    return None, None


def parse_blast_alignment(aligned_ref, aligned_ali):
    ref_ptr, ali_ptr = 0, 0
    insert_counts = [0]
    delete_labels = [1]  # BOS token is always marked as deleted

    while ali_ptr < len(aligned_ali):
        if aligned_ali[ali_ptr] == '-':
            insert_counts[-1] += 1
            ali_ptr += 1
        else:
            if aligned_ref[ali_ptr] == '-':
                delete_labels.append(1)
            else:
                delete_labels.append(0)
                ref_ptr += 1
            ali_ptr += 1
            insert_counts.append(0)
    return insert_counts, delete_labels


for ref_ids, hyp_ids, _, _, ref_raws, ali_raws in train_loader:
    ref_raw, ali_raw = ref_raws[0], ali_raws[0]

    with tempfile.NamedTemporaryFile(delete=False, mode='w') as ref_file, tempfile.NamedTemporaryFile(delete=False, mode='w') as ali_file:
        write_fasta(ref_raw, ref_file.name, "ref")
        write_fasta(ali_raw, ali_file.name, "ali")

        output_file = tempfile.NamedTemporaryFile(delete=False).name
        blastp_cline = NcbiblastpCommandline(
            query=ali_file.name,
            subject=ref_file.name,
            outfmt=5,  # XML format
            out=output_file
        )
        blastp_cline()

    aligned_ref, aligned_ali = parse_blast_xml(output_file)
    if aligned_ref is None or aligned_ali is None:
        print("No alignment found, skipping...")
        os.unlink(ref_file.name)
        os.unlink(ali_file.name)
        os.unlink(output_file)
        continue

    insert_counts, delete_labels = parse_blast_alignment(aligned_ref, aligned_ali)
    insert_logits = torch.tensor(insert_counts).unsqueeze(1).float()
    insert_target = torch.tensor(insert_counts).unsqueeze(1).float()
    delete_logits = torch.tensor(delete_labels).unsqueeze(1).float()
    delete_target = torch.tensor(delete_labels).unsqueeze(1).float()

    metrics = evaluate_insertion_deletion(insert_logits, insert_target, delete_logits, delete_target)
    print(metrics)

    os.unlink(ref_file.name)
    os.unlink(ali_file.name)
    os.unlink(output_file)
    break  # Remove this break to process the entire dataset
