from fastcoref import TrainingArgs, CorefTrainer


args = TrainingArgs(
    output_dir='test-trainer',
    overwrite_output_dir=True,
    model_name_or_path='distilroberta-base',
    device='cuda:2',
    epochs=129,
    logging_steps=100,
    eval_steps=100
)
trainer = CorefTrainer(
    args=args,
    # train_file='/Users/sotmazgin/Desktop/fastcoref/dev.english.jsonlines',
    # dev_file='/Users/sotmazgin/Desktop/fastcoref/dev.english.jsonlines',
    train_file='/home/nlp/shon711/lingmess-coref/prepare_ontonotes/train.english.jsonlines',
    dev_file='/home/nlp/shon711/lingmess-coref/prepare_ontonotes/dev.english.jsonlines'
)
trainer.train()
trainer.evaluate()