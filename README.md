Environment:
1. python=3.10
2. prody: needs to install from github. Package available in ProDy folder.
    cd ProDy
    python setup.py build_ext --inplace --force
    pip install -Ue .
3. pytorch=2.6.0 (pip install torch)
4. notebook
5. biotite
6. tqdm
7. transformers

Align using Toy example dataset:
python align.py   --data_file database/toy_dataset/   --output_file msa_alignments.txt   --model_name_or_path facebook/esm2_t6_8M_UR50D   --align_layer 6   --extraction softmax   --batch_size 1   --num_workers 1
