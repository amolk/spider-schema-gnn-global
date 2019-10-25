# Generate the beam output from the base model
 allennlp predict ../spider-schema-gnn-bert/experiments/bert-spider-full/ ./datasets/spider/dev.json --predictor spider_discriminator --use-dataset-reader --cuda-device=0 --silent --output-file dev_beam.jsonlines --include-package models.semantic_parsing.spider_parser --include-package dataset_readers.spider --include-package predictors.discriminator_dataset_generator --weights-file ../spider-schema-gnn-bert/experiments/bert-spider-full/best.th -o "{\"dataset_reader\":{\"keep_if_unparsable\":true}}"

# Run the reranker and generate the predictions
 python rerank_beam_outputs.py  --input-file dev_beam.jsonlines --output-file predictions.sql --model best_discrim_model_full_spider_retrained.state.pth 

# Evaluate
PYTHONPATH=. python semparse/worlds/evaluate.py --gold datasets/spider/dev_gold.sql --pred predictions.sql --db datasets/spider/database/ --table datasets/spider/tables.json --etype match