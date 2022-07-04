python run.py \
      --output_file=lingmess_predictions.jsonlines \
      --model_name_or_path=lingmess-longformer-final-model-dgx02/model \
      --test_file=prepare_ontonotes/dev.english.jsonlines \
      --eval_split=test \
      --max_tokens_in_batch=5000 \
      --device=cuda:0
