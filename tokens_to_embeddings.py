from transformers import EsmTokenizer, EsmModel
import json
from os import listdir
from os.path import isfile

tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
print(tokenizer.get_vocab())
model = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D')

def tokenize(database):
    path = f"database/{database}/"
    files = [f for f in listdir(path) if isfile(path+"/"+f)]

    count = 0
    for item in listdir(path):
        if isfile(path+"/"+item):
            sequences = {}
            with open(path+item, "r") as file:

                for line in file:

                    filtered = ""
                    split = line.split()
                    #print(split[1])
                    for letter in split[1]:
                        if letter != "-": filtered+=letter

                    if filtered not in sequences.keys():
                        tokenized = tokenizer(filtered,  return_tensors="pt")
                        #print(tokenized)

                        output = model(tokenized.input_ids, tokenized.attention_mask)
                        #print(sequence.last_hidden_state.shape, len(filtered))
                        
                        sequence = output.last_hidden_state.squeeze()
                        count += sequence.shape[0]

                        ## I tried making an actual list of each array in the tensor because I thought the torch.tolist() on the overall tensor was the problem but it was not
                        sequence_list = []
                        for i in range(sequence.shape[0]):
                            sequence_list.append(sequence[i].tolist())
                        
                        sequences[filtered] = sequence_list
    print(count*320)

    out_file = open("sequence_embeddings.json", "a")
    json.dump(sequences, out_file, indent = 4, sort_keys=False)
    out_file.close()

#tokenize("toy_dataset")
