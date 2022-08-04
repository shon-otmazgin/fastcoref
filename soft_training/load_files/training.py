import json
import os
import logging

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from utilities.util import save_all, softXEnt
import wandb

logger = logging.getLogger(__name__)


def train(args, train_batches, model, tokenizer, evaluator):
    """ Train the model """
    t_total = len(train_batches) * args.train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    head_params = ['coref', 'mention', 'antecedent']

    model_decay = [p for n, p in model.named_parameters() if
                   not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
    model_no_decay = [p for n, p in model.named_parameters() if
                      not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
    head_decay = [p for n, p in model.named_parameters() if
                  any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
    head_no_decay = [p for n, p in model.named_parameters() if
                     any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]

    head_learning_rate = args.head_learning_rate if args.head_learning_rate else args.learning_rate
    optimizer_grouped_parameters = [
        {'params': model_decay, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': model_no_decay, 'lr': args.learning_rate, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': head_learning_rate, 'weight_decay': args.weight_decay},
        {'params': head_no_decay, 'lr': head_learning_rate, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total * 0.1, num_training_steps=t_total)

    # using mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.train_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, tr_loss, logging_loss = 0, 0.0, 0.0
    best_f1, best_global_step = -1, -1

    train_iterator = tqdm(range(int(args.train_epochs)), desc="Epoch")
    teacher_logits_dir = os.path.dirname(args.dataset_files['train'])
    for _ in train_iterator:
        epoch_iterator = tqdm(train_batches, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            batch['input_ids'] = torch.tensor(batch['input_ids'], device=args.device)
            batch['attention_mask'] = torch.tensor(batch['attention_mask'], device=args.device)
            if 'leftovers' in batch:
                batch['leftovers']['input_ids'] = torch.tensor(batch['leftovers']['input_ids'], device=args.device)
                batch['leftovers']['attention_mask'] = torch.tensor(batch['leftovers']['attention_mask'], device=args.device)

            keys = [doc_key.replace('/', '_') for doc_key in batch['doc_key']]
            teacher_coref_logits = torch.from_numpy(np.stack([np.load(os.path.join(teacher_logits_dir, k + '_coref_logits.npy'))
                                                              for k in keys], axis=0)).to(args.device)
            topk_1d_indices = torch.from_numpy(np.stack([np.load(os.path.join(teacher_logits_dir, k + '_top_indices.npy'))
                                                         for k in keys], axis=0)).to(args.device)
            model.zero_grad()
            model.train()

            with torch.cuda.amp.autocast():
                outputs = model(batch, topk_1d_indices=topk_1d_indices, return_all_outputs=True)
                span_mask = outputs[0]
                student_coref_logits = outputs[-1]

            loss = softXEnt(teacher_logits=teacher_coref_logits, student_logits=student_coref_logits, span_mask=span_mask)

            tr_loss += loss.item()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scheduler.step()        # Update learning rate schedule
            scaler.update()         # Updates the scale for next iteration
            global_step += 1

            # Log metrics
            if global_step % args.logging_steps == 0:
                loss = (tr_loss - logging_loss) / args.logging_steps
                logger.info(f"\nloss step {global_step}: {loss}")
                wandb.log({'loss': loss}, step=global_step)
                logging_loss = tr_loss

            # Evaluation
            if global_step % args.eval_steps == 0:
                results = evaluator.evaluate(model, prefix=f'step_{global_step}')
                wandb.log(results, step=global_step)

                f1 = results["f1"]
                if f1 > best_f1:
                    best_f1, best_global_step = f1, global_step
                    wandb.run.summary["best_f1"] = best_f1

                    # Save model
                    output_dir = os.path.join(args.output_dir, f'model')
                    save_all(tokenizer=tokenizer, model=model, output_dir=output_dir)
                logger.info(f"best f1 is {best_f1} on global step {best_global_step}")

    with open(os.path.join(args.output_dir, f"best_f1.json"), "w") as f:
        json.dump({"best_f1": best_f1, "best_global_step": best_global_step}, f)

    return global_step, tr_loss / global_step



