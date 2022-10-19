import sys
sys.path.append('/home/nlp/shon711/fast-coref')

import logging
import os
import shutil

from utilities import coref_dataset
import torch
from transformers import AutoConfig, AutoTokenizer

from utilities.consts import SUPPORTED_MODELS
from models.modeling_fcoref import FCorefModel
from models.modeling_s2e import S2E
from models.modeling_lingmess import LingMessModel
from soft_training.training import train
from utilities.eval import Evaluator
from utilities.util import set_seed
from utilities.cli import parse_args
from utilities.collate import PadCollator, DynamicBatchSampler, LeftOversCollator
import wandb

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


def get_model(model_name_or_path, coref_class, args):
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=args.cache_dir)

    model, loading_info = coref_class.from_pretrained(
        model_name_or_path, output_loading_info=True,
        config=config, cache_dir=args.cache_dir, args=args
    )

    if model.base_model_prefix not in SUPPORTED_MODELS:
        raise NotImplementedError(f'Model not supporting {args.model_type}, choose one of {SUPPORTED_MODELS}')

    model.to(args.device)
    for key, val in loading_info.items():
        logger.info(f'{key}: {val}')

    t_params, h_params = [p / 1000000 for p in model.num_parameters()]
    logger.info(f'Parameters: {t_params + h_params:.1f}M, Transformer: {t_params:.1f}M, Head: {h_params:.1f}M')

    return model


def main():
    args = parse_args()

    if args.experiment_name is not None:
        wandb.init(project=args.experiment_name, config=args)

    if args.output_dir is not None:
        if os.path.exists(args.output_dir):
            if args.overwrite_output_dir:
                shutil.rmtree(args.output_dir)
                logger.info(f'--overwrite_output_dir used. directory {args.output_dir} deleted!')
            else:
                raise ValueError(f"Output directory ({args.output_dir}) already exists. Use --overwrite_output_dir to overcome.")
        os.mkdir(args.output_dir)
    else:
        if args.do_train:
            raise ValueError(f"Output directory is required while do_train=True.")
        else:
            if args.output_file is None:
                raise ValueError(f"Output directory or output file is required.")

    # Setup CUDA, GPU & distributed training
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = 1
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True,
                                              add_prefix_space=True, cache_dir=args.cache_dir)

    student = get_model(args.model_name_or_path, FCorefModel, args)

    ffnn_size = args.ffnn_size
    args.ffnn_size = 2048
    teacher = get_model('biu-nlp/lingmess-coref', LingMessModel, args)
    # teacher = get_model('/home/nlp/shon711/fastcoref/trained_longformer', S2E, args)
    args.ffnn_size = ffnn_size

    # load datasets
    dataset, dataset_files = coref_dataset.create(
        tokenizer=tokenizer,
        train_file=args.train_file, dev_file=args.dev_file, test_file=args.test_file,
        cache_dir=args.cache_dir
    )
    args.dataset_files = dataset_files

    student_collator = LeftOversCollator(tokenizer=tokenizer, device=args.device, max_segment_len=args.max_segment_len)
    eval_dataloader = DynamicBatchSampler(
        dataset[args.eval_split],
        collator=student_collator,
        max_tokens=args.max_tokens_in_batch,
        max_segment_len=args.max_segment_len,
        max_doc_len=4096
    )
    evaluator = Evaluator(args=args, eval_dataloader=eval_dataloader)

    # Training
    if args.do_train:
        student_train_sampler = DynamicBatchSampler(
            dataset['train'],
            collator=student_collator,
            max_tokens=args.max_tokens_in_batch,
            max_segment_len=args.max_segment_len,
            max_doc_len=4096
        )
        student_train_batches = coref_dataset.create_batches(
            sampler=student_train_sampler, dataset_files=args.dataset_files, cache_dir=args.cache_dir
        ).shuffle(seed=args.seed)
        logger.info(student_train_batches)

        teacher_collator = PadCollator(tokenizer=tokenizer, device=args.device)
        teacher_train_sampler = DynamicBatchSampler(
            dataset['train'],
            collator=teacher_collator,
            max_tokens=args.max_tokens_in_batch,
            max_segment_len=args.max_segment_len,
            max_doc_len=4096
        )
        teacher_train_batches = coref_dataset.create_batches(
            sampler=teacher_train_sampler, dataset_files=args.dataset_files, cache_dir=args.cache_dir
        ).shuffle(seed=args.seed)
        logger.info(teacher_train_batches)

        global_step, tr_loss = train(args, student_train_batches, teacher_train_batches, student, teacher, tokenizer, evaluator)
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

    # Evaluation
    results = evaluator.evaluate(student)

    # model.push_to_hub("lingmess-coref", organization='biu-nlp', use_temp_dir=True)
    # tokenizer.push_to_hub("lingmess-coref", organization='biu-nlp', use_temp_dir=True)

    return results


if __name__ == "__main__":
    main()
