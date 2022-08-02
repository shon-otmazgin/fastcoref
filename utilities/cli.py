import argparse

MODEL_TYPES = ['longformer', 'roberta']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="allenai/longformer-large-4096",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")

    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Training set file path. should be jsonlines."
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default=None,
        help="Development set file path. should be jsonlines."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Test set file path. should be jsonlines."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Prediction output file path. output will be in jsonlines format."
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--eval_split", type=str, required=True)

    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--head_learning_rate", default=3e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.3, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--ffnn_size", type=int, default=1024)

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every X updates steps.")

    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--max_span_length", type=int, default=30)
    parser.add_argument("--top_lambda", type=float, default=0.4)

    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default='cache')

    parser.add_argument("--max_segment_len", type=int, default=512)
    parser.add_argument("--max_tokens_in_batch", type=int, default=5000)

    # parser.add_argument("--conll_path_for_eval", type=str, default=None)

    args = parser.parse_args()
    return args
