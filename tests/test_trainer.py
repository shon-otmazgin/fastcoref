from fastcoref import TrainingArgs, CorefTrainer


args = TrainingArgs(
    output_dir='test-trainer',
    overwrite_output_dir=True,
    model_name_or_path='distilroberta-base',
    device='cpu',
    epochs=2,
    logging_steps=1,
    eval_steps=2,
    max_tokens_in_batch=30
)
trainer = CorefTrainer(
    args=args,
    train_file='out_test.jsonlines',
)
trainer.train()
trainer.evaluate(test=True)

trainer = CorefTrainer(
    args=args,
    train_file='out_test.jsonlines',
    dev_file='out_test.jsonlines',
    test_file='out_test.jsonlines'
)
trainer.train()
trainer.evaluate(test=True)
