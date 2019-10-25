from pathlib import Path
from transformers import *
import json
from tqdm import tqdm
import argparse
import functools
import torch
import torch.nn.functional as F


cuda_is_available = torch.cuda.is_available()
thresh = 0.10

def cuda(obj):
    if cuda_is_available:
        return obj.cuda()
    else:
        return obj

def load_beam_outputs(filename):
  filename = Path(filename)
  data = []
  with filename.open('r') as f:
    for line in f:
      data.append(json.loads(line))
  return data

# Creates the input for BERT:
# Returns the tokenized bert input [CLS] <tokenized nlu> [SEP] col1 [SEP] col2 ...[SEP]
# and the start and end locations in the tokenized input for each of the columns
def create_tokenized_input(tokenizer, nlu, columns, tables):
    tokens = []
    tokens += tokenizer.tokenize('[CLS]')
    tokens += tokenizer.tokenize(nlu)
    tokens += tokenizer.tokenize('[SEP]')
    column_locations = []
    for column in columns:
        column_str = '.'.join(column)
        tokens += tokenizer.tokenize(column_str)
        tokens += tokenizer.tokenize('[SEP]')
    for table in tables:
        tokens += tokenizer.tokenize(table)
        tokens += tokenizer.tokenize('[SEP]')
    return tokens, tokenizer.convert_tokens_to_ids(tokens)

def load_model(model_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(model_path))
    model = cuda(model)
    model.eval()
    return model, tokenizer

def run_instance(bert_model, tokenizer, utterance, instance):
    tokens, encoded_tokens = create_tokenized_input(tokenizer, utterance, instance['columns_used'], instance['tables_used'])
    encoded_tokens = cuda(torch.tensor(encoded_tokens).unsqueeze(0))
    with torch.no_grad():
        outputs = bert_model(encoded_tokens)
    logits = outputs[0]
    preds = F.softmax(logits, dim=1)
    pred_index = torch.argmax(preds[0])  # Remove batch
    return preds


def cmp(x, y):
    """
    Replacement for built-in function cmp that was removed in Python 3
    Compare the two objects x and y and return an integer according to
    the outcome. The return value is negative if x < y, zero if x == y
    and strictly positive if x > y.
    """

    return (x > y) - (x < y)

def compare(thresh, a, b):
    a = a['score']
    b = b['score'] + thresh
    return cmp(a, b)

def get_best_prediction(sample, model, tokenizer):
    if 'utterance' not in sample or 'instances' not in sample or len(sample['instances']) == 0:
        return None

    compare_f = functools.partial(compare, thresh)
    instances = sample['instances']
    myinstances = []
    for instance in instances[:10]:
        instance = instance.copy()
        pred = run_instance(model, tokenizer, sample['utterance'], instance)
        instance['score'] = pred[0][1].item()
        myinstances.append(instance)

    pred_instance = list(reversed(sorted(myinstances, key=functools.cmp_to_key(compare_f))))[0]
    return pred_instance['sql_query']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', dest='input_path', type=str)
    parser.add_argument('--output-file', dest='output_path', type=str)
    parser.add_argument('--model', dest='model_path', type=str)

    args = parser.parse_args()

    beam_outputs = load_beam_outputs(args.input_path)

    model, tokenizer = load_model(args.model_path)

    with open(args.output_path, 'w+') as output_file:
        for sample in tqdm(beam_outputs):
            sql = get_best_prediction(sample, model, tokenizer)
            if sql is None:
                output_file.write('NO PREDICTION')
            else:
                output_file.write(sql)
            output_file.write('\n')

