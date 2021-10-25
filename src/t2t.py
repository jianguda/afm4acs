import os
import re
import csv
import time
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
import torch
import random
import numpy as np
import argparse
from metric import Metric
from torch.utils.data import SequentialSampler, BatchSampler
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForLanguageModeling,
    RobertaForMaskedLM,
    GPT2LMHeadModel,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


@dataclass
class Option:
    # hyper
    epoch_num: int
    # data label
    dataset_label: str  # basts gh sit so
    # model label
    load_label: str  # which model to load
    dump_label: str  # dump to which model


@dataclass
class Job:
    """
    the alias are from https://ruder.io/recent-advances-lm-fine-tuning/
    AF (adaptive_finetune) is specialized for target domain/task, where we
        continuously pretrain the encoder and decoder models separately
    BF (behavioral_finetune) is specialized for target domain/task, where we
        finetune the encoder-decoder model for the intermediate proxy tasks
    CF (common_finetune) is specialized for downstream dataset and target task
    """
    # continuous pretraining
    af_epoch: int
    af_data: str
    af_mode: str
    # intermediate finetuning
    bf_epoch: int
    bf_data: str
    bf_task: str
    # the usual finetuning
    cf_epoch: int
    cf_data: str


class DataLoader:
    @staticmethod
    def _load_basts_data(lang, split):
        assert lang in ('java', 'python')
        assert split in ('train', 'valid', 'test')
        data_dir = Path.cwd().parent / 'data' / 'basts' / lang
        with open(data_dir / split / f'{split}.token.code') as file:
            sources = [line.strip().lower() for line in file]
        with open(data_dir / split / f'{split}.token.nl') as file:
            targets = [line.strip().lower() for line in file]
        return Dataset.from_dict({'snippets': sources, 'comments': targets})

    @staticmethod
    def _load_sit_data(lang, split):
        assert lang in ('java', 'python')
        assert split in ('train', 'valid', 'test')
        data_dir = Path.cwd().parent / 'data' / 'sit' / lang
        with open(data_dir / f'{split}.token.code') as file:
            sources = [line.strip().lower() for line in file]
        with open(data_dir / f'{split}.token.nl') as file:
            targets = [line.strip().lower() for line in file]
        return Dataset.from_dict({'snippets': sources, 'comments': targets})

    @staticmethod
    def _load_gh_data(lang, split):
        assert lang in ('java', 'python')
        # assert split in ('train', 'validation', 'test')
        assert split in ('train', 'valid')
        # we merge <train + validation> as <new train> and see <test> as <new test>
        if split == 'valid':
            split_dataset = load_dataset("code_x_glue_ct_code_to_text", lang)['test']
            split_dataset = split_dataset.map(
                lambda sample: {
                    # 'code_tokens' is much better than 'code'
                    'snippets': str(' '.join(sample['code_tokens'])).lower(),
                    # 'docstring_tokens' is much better than 'docstring'
                    'comments': str(' '.join(sample['docstring_tokens'])).lower(),
                }
            )
        else:
            train_dataset = load_dataset("code_x_glue_ct_code_to_text", lang)['train']
            train_dataset = train_dataset.map(
                lambda sample: {
                    # 'code_tokens' is much better than 'code'
                    'snippets': str(' '.join(sample['code_tokens'])).lower(),
                    # 'docstring_tokens' is much better than 'docstring'
                    'comments': str(' '.join(sample['docstring_tokens'])).lower(),
                }
            )
            valid_dataset = load_dataset("code_x_glue_ct_code_to_text", lang)['validation']
            valid_dataset = valid_dataset.map(
                lambda sample: {
                    # 'code_tokens' is much better than 'code'
                    'snippets': str(' '.join(sample['code_tokens'])).lower(),
                    # 'docstring_tokens' is much better than 'docstring'
                    'comments': str(' '.join(sample['docstring_tokens'])).lower(),
                }
            )
            split_dataset = Dataset.from_dict({
                'snippets': train_dataset['snippets'] + valid_dataset['snippets'],
                'comments': train_dataset['comments'] + valid_dataset['comments'],
            })
        return split_dataset

    @staticmethod
    def _load_so_data(lang, split):
        assert lang in ('java', 'python')
        assert split in ('train', 'valid')
        split = 'val' if split == 'valid' else split
        # split: train / val
        data_dir = Path.cwd().parent / 'data' / 'so' / 'pair' / lang
        with open(data_dir / f'{split}.src') as file:
            sources = [line.strip().lower() for line in file]
        with open(data_dir / f'{split}.tgt') as file:
            targets = [line.strip().lower() for line in file]
        return Dataset.from_dict({'snippets': sources, 'comments': targets})

    @staticmethod
    def load_data(label, lang, split, dryrun=False):
        assert label in ('basts', 'sit', 'gh', 'so')
        assert lang in ('java', 'python')
        assert split in ('train', 'valid', 'test')
        # split: train / valid / test
        # dataset: basts sit gh so
        if label == 'basts':
            dataset = DataLoader._load_basts_data(lang, split)
        elif label == 'sit':
            dataset = DataLoader._load_sit_data(lang, split)
        elif label == 'gh':
            dataset = DataLoader._load_gh_data(lang, split)
        elif label == 'so':
            dataset = DataLoader._load_so_data(lang, split)
        else:
            raise NotImplementedError
        # print(dataset['snippets'][:3])
        # print(dataset['comments'][:3])
        if dryrun:
            dataset = dataset.select(range(800)) if split == 'train' else dataset.select(range(100))
        return dataset

    @staticmethod
    def clean_dataset(dataset):
        def _clean_code(code_data):
            code_tokens = code_data.split()
            code_tokens = list(filter(lambda x: x.isalnum(), code_tokens))
            return ' '.join(code_tokens)

        def _clean_text(text_data):
            text_tokens = text_data.split()
            text_tokens = list(filter(lambda x: x.isalnum(), text_tokens))
            return ' '.join(text_tokens)

        cleaned_dataset = dataset.map(
            lambda sample: {
                'snippets': _clean_code(sample['snippets']),
                'comments': _clean_text(sample['comments']),
            }
        )
        return cleaned_dataset

    @staticmethod
    def refine_dataset(dataset, task):
        assert task in ('ca', 'ce', 'ci')

        def _refine(src, tgt):
            src_tokens = src.split()
            tgt_tokens = tgt.split()
            common_tokens = set(src_tokens) & set(tgt_tokens)
            if task == 'ca':
                return ' '.join(['@+@' if token in common_tokens else '@-@' for token in tgt_tokens])
            elif task == 'ce':
                return ' '.join(['@+@' if token in common_tokens else token for token in tgt_tokens])
            elif task == 'ci':
                return ' '.join(['@-@' if token not in common_tokens else token for token in tgt_tokens])
            else:
                raise NotImplementedError

        refined_dataset = dataset.map(
            lambda sample: {
                'snippets': sample['snippets'],
                'comments': _refine(sample['snippets'], sample['comments']),
            }
        )
        return refined_dataset


class T2T:
    @staticmethod
    def define_model(load_path=None):
        encoder_tag = args.encoder
        decoder_tag = args.decoder
        encoder_url = tag_table[encoder_tag]
        decoder_url = tag_table[decoder_tag]

        if args.orz == 'adamo':
            model_constructor = EncoderDecoderModel
        elif args.orz == 'noisy':
            from noisy_model import NoisyEncoderDecoderModel
            model_constructor = NoisyEncoderDecoderModel
        else:
            raise NotImplementedError

        if load_path is not None:
            model = model_constructor.from_pretrained(load_path)
        elif job.af_epoch == 0:
            model = model_constructor.from_encoder_decoder_pretrained(
                encoder_url, decoder_url, gaussian=args.gaussian, impulsive=args.impulsive)
        else:
            exp_prefix = ('_' if args.dryrun else '') + args.mark
            encoder_label = f'{exp_prefix}{args.lang}_{encoder_tag}_af+{job.af_data}'
            decoder_label = f'{exp_prefix}{args.lang}_{decoder_tag}_af+{job.af_data}'
            # use 'relative path' instead of 'absolute path'
            encoder_path = encoder_url if job.af_mode == 'clm' else f'../models/pretrained{args.power}/{encoder_label}/'
            decoder_path = decoder_url if job.af_mode == 'mlm' else f'../models/pretrained{args.power}/{decoder_label}/'
            # encoder_path = Path.cwd().parent / 'models' / 'pretrained' / encoder_label
            # decoder_path = Path.cwd().parent / 'models' / 'pretrained' / decoder_label
            print(encoder_path)
            print(decoder_path)
            print('+' * 16)
            try:
                model = model_constructor.from_encoder_decoder_pretrained(encoder_path, decoder_path)
                print('load checkpoints smoothly')
                logger.info('load checkpoints smoothly')
            except Exception as e:
                print('!' * 9)
                print(e)
                model = model_constructor.from_encoder_decoder_pretrained(encoder_url, decoder_url)

        if args.orz == 'freeze':  # seems not so good
            for name, parameter in model.encoder.base_model.named_parameters():
                print(name, parameter.size())
                if 'intermediate.' in name:  # FeedForward
                    continue
                if 'LayerNorm.' in name:  # LayerNorm
                    continue
                for layer_id in range(12):  # or 11 or 10
                    if f'layer.{layer_id}.' in name:
                        parameter.requires_grad = False
                        break
            for name, parameter in model.decoder.base_model.named_parameters():
                print(name, parameter.size())
                if 'mlp.' in name:  # FeedForward
                    continue
                if 'ln_1.' in name or 'ln_2.' in name:  # LayerNorm
                    continue
                for layer_id in range(12):  # or 11 or 10
                    if f'h.{layer_id}.' in name:
                        parameter.requires_grad = False
                        break

        model.config.min_length = 4
        model.config.max_length = 64
        model.config.vocab_size = model.config.encoder.vocab_size
        tokenizer = AutoTokenizer.from_pretrained(encoder_url, model_max_length=input_max_length)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        return model, tokenizer

    # for af
    @staticmethod
    def train_model(option, tokenizer, model, data_type, objective):
        model.to(device)
        model.train()

        mlm = True if objective == 'mlm' else False
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm, mlm_probability=0.15)
        train_dataset = DataLoader.load_data(option.dataset_label, args.lang, 'train', args.dryrun)
        valid_dataset = DataLoader.load_data(option.dataset_label, args.lang, 'valid', args.dryrun)
        # train_dataset = DataLoader.clean_dataset(train_dataset)
        # valid_dataset = DataLoader.clean_dataset(valid_dataset)

        def _map_to_trainer_inputs(batch):
            # Tokenizer will automatically set [BOS] <text> [EOS]
            max_length = {'snippets': input_max_length, 'comments': 64}[data_type]
            inputs = tokenizer(batch[data_type], padding='max_length', truncation=True, max_length=max_length)
            batch['input_ids'] = inputs.input_ids
            batch['labels'] = inputs.input_ids.copy()
            batch['labels'] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch['labels']
            ]

            assert all([len(x) == max_length for x in inputs.input_ids])
            return batch

        train_dataset = train_dataset.map(
            _map_to_trainer_inputs,
            batched=True,
            batch_size=128,
            remove_columns=['snippets', 'comments'],
        )
        train_dataset.set_format(
            type='torch',
            columns=['input_ids', 'labels'],
        )
        valid_dataset = valid_dataset.map(
            _map_to_trainer_inputs,
            batched=True,
            batch_size=128,
            remove_columns=['snippets', 'comments'],
        )
        valid_dataset.set_format(
            type='torch',
            columns=['input_ids', 'labels'],
        )

        # instantiate trainer
        dump_path = Path.cwd().parent / 'models' / f'checkpoints{args.power}' / option.dump_label
        training_args = TrainingArguments(
            seed=42,
            report_to='none',
            # save_total_limit=1,
            save_strategy='epoch',
            output_dir=str(dump_path),
            # load_best_model_at_end=True,
            per_device_train_batch_size=32 if mlm else 64,
            per_device_eval_batch_size=64 if mlm else 128,
            num_train_epochs=option.epoch_num,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )
        print('+++++++++ model training +++++++++')
        logger.info('+++++++++ model training +++++++++')
        try:
            trainer.train(resume_from_checkpoint=True)
        except Exception as e:
            print('!' * 9)
            print(e)
            trainer.train()

    # for bf & cf
    @staticmethod
    def tune_model(option, tokenizer, model):
        def _map_to_encoder_decoder_inputs(batch):
            # Tokenizer will automatically set [BOS] <text> [EOS]
            inputs = tokenizer(batch['snippets'], padding='max_length', truncation=True, max_length=input_max_length)
            outputs = tokenizer(batch['comments'], padding='max_length', truncation=True, max_length=64)
            batch['input_ids'] = inputs.input_ids
            batch['attention_mask'] = inputs.attention_mask
            batch['decoder_input_ids'] = outputs.input_ids
            batch['labels'] = outputs.input_ids.copy()
            batch['labels'] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch['labels']
            ]
            batch['decoder_attention_mask'] = outputs.attention_mask

            assert all([len(x) == input_max_length for x in inputs.input_ids])
            assert all([len(x) == 64 for x in outputs.input_ids])
            return batch

        model.to(device)
        model.train()
        # load dataset but distinguish bf and cf
        train_dataset = DataLoader.load_data(option.dataset_label, args.lang, 'train', args.dryrun)
        valid_dataset = DataLoader.load_data(option.dataset_label, args.lang, 'valid', args.dryrun)
        if '_bf' in option.dump_label and '_cf' not in option.dump_label:
            train_dataset = DataLoader.refine_dataset(train_dataset, job.bf_task)
            valid_dataset = DataLoader.refine_dataset(valid_dataset, job.bf_task)
        # map the training dataset
        train_dataset = train_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=256,
            remove_columns=['snippets', 'comments'],
        )
        train_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'],
        )
        # map the validation dataset
        valid_dataset = valid_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=256,
            remove_columns=['snippets', 'comments'],
        )
        valid_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'],
        )
        # instantiate trainer
        if job.af_epoch > 0:
            dump_path = Path.cwd().parent / 'models' / f'checkpoints{args.power}' / option.dump_label
        else:
            dump_path = Path.cwd().parent / 'models' / 'checkpoints' / option.dump_label
        training_args = Seq2SeqTrainingArguments(
            seed=42,
            report_to='none',
            # save_total_limit=10,
            save_strategy='epoch',
            output_dir=str(dump_path),
            predict_with_generate=True,
            # load_best_model_at_end=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=option.epoch_num,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )
        print('+++++++++ model tuning +++++++++')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        logger.info('+++++++++ model tuning +++++++++')
        logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        try:
            trainer.train(resume_from_checkpoint=True)
        except Exception as e:
            print('!' * 9)
            print(e)
            trainer.train()

    @staticmethod
    def infer_model(tokenizer, model, file_path=None):
        model.to(device)
        model.eval()

        test_dataset = DataLoader.load_data(job.cf_data, args.lang, 'test', args.dryrun)
        predictions = list()
        if args.dryrun:
            sampler = SequentialSampler(test_dataset['snippets'][:100])
        else:
            sampler = SequentialSampler(test_dataset['snippets'])
        chunked_indexes = BatchSampler(sampler, batch_size=64, drop_last=False)
        print('+++++++++ model inferring +++++++++')
        logger.info('+++++++++ model inferring +++++++++')
        for indexes in chunked_indexes:
            chunked_data = [test_dataset['snippets'][index] for index in indexes]
            chunked_input_dict = tokenizer(
                chunked_data,
                max_length=input_max_length, truncation=True,
                padding='max_length', return_tensors='pt',
            )
            chunked_output_ids = model.generate(
                input_ids=chunked_input_dict['input_ids'].to(device),
                attention_mask=chunked_input_dict['attention_mask'].to(device),
                num_beams=5,
                min_length=model.config.min_length,
                max_length=model.config.max_length,
            )
            chunked_predictions = tokenizer.batch_decode(chunked_output_ids,
                                                         skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False)
            predictions.extend(chunked_predictions)

        references = test_dataset['comments']
        if args.dryrun:
            references = references[:100]
        print(predictions[:5])
        print(references[:5])
        if file_path is not None and not (args.dryrun or args.fastrun):
            with open(file_path, 'w') as file:
                for datum in predictions:
                    file.write(f'{datum}\n')
        Metric.report(predictions, references)

    @staticmethod
    def run_pretrain(option, model_url, model_path, data_type, objective):
        if model_path.exists():
            print('return directly at run_pretrain')
            logger.info('return directly at run_pretrain')
            return
        if objective == 'mlm':
            assert 'bert' in args.encoder
            # with pretrained weights
            model = RobertaForMaskedLM.from_pretrained(model_url)
            # without pretrained weights
            # model = AutoModelForMaskedLM.from_pretrained(model_url)
        elif objective == 'clm':
            assert 'gpt2' in args.decoder
            # with pretrained weights
            model = GPT2LMHeadModel.from_pretrained(model_url)
            # without pretrained weights
            # model = AutoModelForCausalLM.from_pretrained(model_url)
        else:
            raise NotImplementedError

        tokenizer = AutoTokenizer.from_pretrained(model_url, model_max_length=input_max_length)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        T2T.train_model(option, tokenizer, model, data_type, objective)
        try:
            model.base_model.save_pretrained(model_path)
        except Exception as e:
            print('!' * 9)
            print(e)

    @staticmethod
    def run_pipeline(option, load_path, dump_path):
        # load model
        try:
            model, tokenizer = T2T.define_model()
            model = EncoderDecoderModel.from_pretrained(load_path)
            # tokenizer = AutoTokenizer.from_pretrained(load_path, model_max_length=input_max_length)
        except Exception as e:
            print(e)
            model, tokenizer = T2T.define_model()
        # model, tokenizer = T2T.define_model(load_path) if load_path.exists() else T2T.define_model()
        print('load the encoder-decoder model')
        logger.info('load the encoder-decoder model')
        # train the model
        print('tune the encoder-decoder model')
        T2T.tune_model(option, tokenizer, model)
        # save model
        print('save the encoder-decoder model')
        logger.info('save the encoder-decoder model')
        model.save_pretrained(dump_path)
        tokenizer.save_pretrained(dump_path)


class Mission:
    @staticmethod
    def train_t2t():
        exp_prefix = ('_' if args.dryrun else '') + args.mark
        dump_label = f'{exp_prefix}{args.orz}_{args.lang}'
        if args.orz == 'noisy':
            dump_label += f'_g{args.gaussian}_i{args.impulsive}'
        if job.af_epoch == job.bf_epoch == job.cf_epoch == 0:
            pass
        else:
            if job.af_epoch > 0:
                dump_label = f'{dump_label}_af+{job.af_data}+{job.af_mode}'
                encoder_tag = args.encoder
                decoder_tag = args.decoder
                encoder_url = tag_table[encoder_tag]
                decoder_url = tag_table[decoder_tag]
                encoder_label = f'{exp_prefix}{args.lang}_{encoder_tag}_af+{job.af_data}'
                decoder_label = f'{exp_prefix}{args.lang}_{decoder_tag}_af+{job.af_data}'
                encoder_path = Path.cwd().parent / 'models' / f'pretrained{args.power}' / encoder_label
                decoder_path = Path.cwd().parent / 'models' / f'pretrained{args.power}' / decoder_label
                if job.af_mode in ['mlm', 'both']:
                    # pretrain the encoder model on snippets with the mlm objective
                    option = Option(job.af_epoch, job.af_data, '', encoder_label)
                    T2T.run_pretrain(option, encoder_url, encoder_path, 'snippets', 'mlm')
                if job.af_mode in ['clm', 'both']:
                    # pretrain the decoder model on comments with the clm objective
                    option = Option(job.af_epoch, job.af_data, '', decoder_label)
                    T2T.run_pretrain(option, decoder_url, decoder_path, 'comments', 'clm')
            if job.bf_epoch > 0:
                load_label = f'{dump_label}'
                dump_label = f'{dump_label}_bf+{job.bf_data}+{job.bf_task}'
                option = Option(job.bf_epoch, job.bf_data, load_label, dump_label)
                if job.af_epoch > 0:
                    load_path = Path.cwd().parent / 'models' / f'pretrained{args.power}' / option.load_label
                    dump_path = Path.cwd().parent / 'models' / f'pretrained{args.power}' / option.dump_label
                else:
                    load_path = Path.cwd().parent / 'models' / 'pretrained' / option.load_label
                    dump_path = Path.cwd().parent / 'models' / 'pretrained' / option.dump_label
                T2T.run_pipeline(option, load_path, dump_path)
            if job.cf_epoch > 0:
                load_label = f'{dump_label}'
                dump_label = f'{dump_label}_cf+{job.cf_data}'
                option = Option(job.cf_epoch, job.cf_data, load_label, dump_label)
                if job.af_epoch > 0:
                    load_path = Path.cwd().parent / 'models' / f'pretrained{args.power}' / option.load_label
                    dump_path = Path.cwd().parent / 'models' / f'pretrained{args.power}' / option.dump_label
                else:
                    load_path = Path.cwd().parent / 'models' / 'pretrained' / option.load_label
                    dump_path = Path.cwd().parent / 'models' / 'pretrained' / option.dump_label
                T2T.run_pipeline(option, load_path, dump_path)

    @staticmethod
    def evaluate_t2t():
        exp_prefix = ('_' if args.dryrun else '') + args.mark
        dump_label = f'{exp_prefix}{args.orz}_{args.lang}'
        if args.orz == 'noisy':
            dump_label += f'_g{args.gaussian}_i{args.impulsive}'
        if job.af_epoch > 0:
            file_dir = Path.cwd().parent / 'results' / f'{args.orz}{args.power}'
        else:
            file_dir = Path.cwd().parent / 'results' / args.orz
        file_dir.mkdir(parents=True, exist_ok=True)
        if job.af_epoch == job.bf_epoch == job.cf_epoch == 0:
            file_path = file_dir / f'{dump_label}_{job.cf_data}_0shot.hyp'
            model, tokenizer = T2T.define_model()
            T2T.infer_model(tokenizer, model, file_path)
        else:
            if job.af_epoch > 0:  # af
                dump_label = f'{dump_label}_af+{job.af_data}+{job.af_mode}'
            if job.bf_epoch > 0:  # bf
                dump_label = f'{dump_label}_bf+{job.bf_data}+{job.bf_task}'
            if job.cf_epoch > 0:  # cf
                dump_label = f'{dump_label}_cf+{job.cf_data}'

            print('*' * 16)
            print('# run evaluation on the dumped model')
            file_path = file_dir / f'{dump_label}.hyp'
            if job.af_epoch > 0:
                load_path = Path.cwd().parent / 'models' / f'pretrained{args.power}' / dump_label
            else:
                load_path = Path.cwd().parent / 'models' / 'pretrained' / dump_label
            model, tokenizer = T2T.define_model(load_path)
            T2T.infer_model(tokenizer, model, file_path)

            # print('*' * 16)
            # print('# run evaluation on the checkpoints')
            # if job.af_epoch > 0:
            #     model_path = Path.cwd().parent / 'models' / f'checkpoints{args.power}' / dump_label
            # else:
            #     model_path = Path.cwd().parent / 'models' / 'checkpoints' / dump_label
            # print(model_path)
            # logger.info(model_path)
            # checkpoint_paths = [it for it in model_path.iterdir() if it.is_dir()]
            # checkpoint_paths.sort(key=lambda x: int(re.findall(r'\d+', str(x))[-1]))
            # # assert job.cf_epoch <= len(checkpoint_paths)
            # length = min(job.cf_epoch, len(checkpoint_paths))
            # for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
            #     if args.wholerun:
            #         if index > length:
            #             continue
            #     elif args.highrun:
            #         if index % 4 > 0 or index > length:
            #             continue
            #     else:
            #         if index != length:
            #             continue
            #     model, tokenizer = T2T.define_model(checkpoint_path)
            #     T2T.infer_model(tokenizer, model)
            #     timestamp = os.path.getmtime(checkpoint_path)
            #     time_str = datetime.datetime.fromtimestamp(timestamp)
            #     print(f'{time_str} Epoch{index}')
            #     logger.info(f'{time_str} Epoch{index}')

    @staticmethod
    def review_t2t():
        references_manager = dict()
        for data_value in ['basts', 'sit']:
            for lang_value in ['java', 'python']:
                identifier = data_value + ':' + lang_value
                references = DataLoader.load_data(data_value, lang_value, 'test')['comments']
                references_manager.update({identifier: references})

        evaluations = list()
        prediction_dir = Path().cwd().parent / 'results'
        print('ready to compute the scores')
        for prediction_folder in prediction_dir.iterdir():
            if not prediction_folder.is_dir():
                continue
            group_value = str(prediction_folder.name)
            for prediction_file in prediction_folder.iterdir():
                if not prediction_file.is_file():
                    continue
                tag_value = str(prediction_file.stem)
                data_value = 'basts' if 'basts' in tag_value else 'sit'
                lang_value = 'java' if 'java' in tag_value else 'python'
                information = [group_value, data_value, lang_value, tag_value]
                with open(prediction_file) as file:
                    lines = file.readlines()
                predictions = [line.strip() for line in lines]
                references = references_manager[data_value + ':' + lang_value]
                assert len(predictions) == len(references)
                scores = Metric.report(predictions, references)
                evaluation = information + list(scores)
                evaluations.append(evaluation)
                print(f'completed {group_value}:{tag_value}')

        print('ready to collect the scores')
        csv_file = prediction_dir / 'scores.csv'
        evaluations = sorted(evaluations, key=lambda x: x[:4])
        header = ['GROUP', 'DATA', 'LANG', 'TAG', 'C-BLEU', 'S-BLEU', 'METEOR', 'ROUGE']
        with open(csv_file, 'w', encoding='utf-8') as file:
            writer = csv.DictWriter(file, header)
            writer.writeheader()
            for evaluation in evaluations:
                writer.writerow(dict(zip(header, evaluation)))


def stats():
    for label in ('basts', 'sit', 'gh'):
        for lang in ('java', 'python'):
            for split in ('train', 'valid', 'test'):
                if label == 'gh' and split == 'test':
                    continue
                dataset = DataLoader.load_data(label, lang, split)
                snippets_length = len(dataset['snippets'])
                comments_length = len(dataset['comments'])
                assert snippets_length == comments_length
                length = snippets_length
                print(f'{label}_{lang}_{split}: {length}')


if __name__ == '__main__':
    # logger
    # Gets or creates a logger
    logger = logging.getLogger(__name__)
    # set log level
    logger.setLevel(logging.INFO)
    # define file handler and set formatter
    # file_handler = logging.FileHandler('logfile.log')
    # formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    # file_handler.setFormatter(formatter)
    # add file handler to logger
    # logger.addHandler(file_handler)
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--power', default=1, type=int, help='[1] 2 3 ...')
    parser.add_argument('--gaussian', default=0.0, type=float, help='[0] 0.09 ...')
    parser.add_argument('--impulsive', default=0.0, type=float, help='[0] 0.01 ...')
    parser.add_argument('--orz', default='adamo', type=str, help='[adamo] noisy freeze')
    parser.add_argument('--job', default='0shot', type=str, help='[0shot] basic ...')
    parser.add_argument('--lang', default='java', type=str, help='[java] python')
    parser.add_argument('--mission', default='full', type=str, help='[full] train evaluate review')
    parser.add_argument('--encoder', default='codebert', type=str, help='roberta [codebert]')
    parser.add_argument('--decoder', default='gpt2', type=str, help='[gpt2]')
    # parser.add_argument('--af_epoch', default=0, type=int, help='[0] 10 ...')
    # parser.add_argument('--af_data', default='gh', type=str, help='[gh] sit')
    # parser.add_argument('--af_mode', default='both', type=str, help='[both] mlm clm')
    # parser.add_argument('--bf_epoch', default=0, type=int, help='[0] 10 ...')
    # parser.add_argument('--bf_data', default='gh', type=str, help='[gh] sit')
    # parser.add_argument('--bf_task', default='ce', type=str, help='[ce] ci')
    # parser.add_argument('--cf_epoch', default=0, type=int, help='[0] 10 ...')
    # parser.add_argument('--cf_data', default='sit', type=str, help='basts [sit]')
    parser.add_argument('--mark', default='', type=str, help='[] ...')
    parser.add_argument('--dryrun', default=False, action='store_true', help='run on a tiny dataset')
    parser.add_argument('--fastrun', default=False, action='store_true', help='run with a small epoch number')
    parser.add_argument('--highrun', default=False, action='store_true', help='run evaluation on more checkpoints')
    parser.add_argument('--wholerun', default=False, action='store_true', help='run evaluation on whole checkpoints')
    args = parser.parse_args()
    print(args)
    logger.info(args)

    # tables
    tag_table = {
        'roberta': 'roberta-base',
        'codebert': 'microsoft/codebert-base',
        'gpt2': 'gpt2',
    }
    input_max_length = {'java': 320, 'python': 256}[args.lang]
    c_epoch_basts = {'java': 8, 'python': 17}[args.lang]  # okay
    a_epoch_mlm_gh = {'java': 20, 'python': 16}[args.lang] * args.power  # okay
    a_epoch_clm_gh = {'java': 93, 'python': 58}[args.lang] * args.power  # okay
    a_epoch_mlm_sit = {'java': 48, 'python': 72}[args.lang] * args.power  # okay
    a_epoch_clm_sit = {'java': 216, 'python': 276}[args.lang] * args.power  # okay
    b_epoch_ca_gh = {'java': 20, 'python': 15}[args.lang]  # okay
    b_epoch_ce_gh = {'java': 18, 'python': 14}[args.lang]  # okay
    b_epoch_ci_gh = {'java': 18, 'python': 14}[args.lang]  # okay
    b_epoch_ca_sit = {'java': 45, 'python': 66}[args.lang]  # okay
    b_epoch_ce_sit = {'java': 45, 'python': 63}[args.lang]  # okay
    b_epoch_ci_sit = {'java': 45, 'python': 63}[args.lang]  # okay
    if args.fastrun:
        c_epoch_sit = {'java': 1, 'python': 1}[args.lang]  # okay
    else:
        c_epoch_sit = {'java': 48, 'python': 60}[args.lang]  # okay
    job_table = {
        # BASTS 0shot
        '0shot4basts': Job(0, 'gh', 'both', 0, 'gh', 'ci', 0, 'basts'),
        # BASTS basic
        'basic4basts': Job(0, 'gh', 'both', 0, 'gh', 'ci', c_epoch_basts, 'basts'),
        # SIT 0shot
        '0shot': Job(0, 'gh', 'both', 0, 'gh', 'ci', 0, 'sit'),
        # SIT basic
        'basic': Job(0, 'gh', 'both', 0, 'gh', 'ci', c_epoch_sit, 'sit'),
        # SIT cp
        # 'cp-da-mlm': Job(a_epoch_mlm_gh, 'gh', 'mlm', 0, 'gh', 'ci', 0, 'sit'),
        # 'cp-da-clm': Job(a_epoch_clm_gh, 'gh', 'clm', 0, 'gh', 'ci', 0, 'sit'),
        # 'cp-ta-mlm': Job(a_epoch_mlm_sit, 'sit', 'mlm', 0, 'gh', 'ci', 0, 'sit'),
        # 'cp-ta-clm': Job(a_epoch_clm_sit, 'sit', 'clm', 0, 'gh', 'ci', 0, 'sit'),
        'cp-da-mlm': Job(a_epoch_mlm_gh, 'gh', 'mlm', 0, 'gh', 'ci', c_epoch_sit, 'sit'),
        'cp-da-clm': Job(a_epoch_clm_gh, 'gh', 'clm', 0, 'gh', 'ci', c_epoch_sit, 'sit'),
        'cp-ta-mlm': Job(a_epoch_mlm_sit, 'sit', 'mlm', 0, 'gh', 'ci', c_epoch_sit, 'sit'),
        'cp-ta-clm': Job(a_epoch_clm_sit, 'sit', 'clm', 0, 'gh', 'ci', c_epoch_sit, 'sit'),
        # no need to train AF again
        'cp-da-both': Job(1, 'gh', 'both', 0, 'gh', 'ci', c_epoch_sit, 'sit'),
        'cp-ta-both': Job(1, 'sit', 'both', 0, 'gh', 'ci', c_epoch_sit, 'sit'),
        # SIT if
        # 'if-da-ca': Job(0, 'gh', 'both', b_epoch_ca_gh, 'gh', 'ca', 0, 'sit'),
        # 'if-da-ce': Job(0, 'gh', 'both', b_epoch_ce_gh, 'gh', 'ce', 0, 'sit'),
        # 'if-da-ci': Job(0, 'gh', 'both', b_epoch_ci_gh, 'gh', 'ci', 0, 'sit'),
        # 'if-ta-ca': Job(0, 'gh', 'both', b_epoch_ca_sit, 'sit', 'ca', 0, 'sit'),
        # 'if-ta-ce': Job(0, 'gh', 'both', b_epoch_ce_sit, 'sit', 'ce', 0, 'sit'),
        # 'if-ta-ci': Job(0, 'gh', 'both', b_epoch_ci_sit, 'sit', 'ci', 0, 'sit'),
        'if-da-ca': Job(0, 'gh', 'both', b_epoch_ca_gh, 'gh', 'ca', c_epoch_sit, 'sit'),
        'if-da-ce': Job(0, 'gh', 'both', b_epoch_ce_gh, 'gh', 'ce', c_epoch_sit, 'sit'),
        'if-da-ci': Job(0, 'gh', 'both', b_epoch_ci_gh, 'gh', 'ci', c_epoch_sit, 'sit'),
        'if-ta-ca': Job(0, 'gh', 'both', b_epoch_ca_sit, 'sit', 'ca', c_epoch_sit, 'sit'),
        'if-ta-ce': Job(0, 'gh', 'both', b_epoch_ce_sit, 'sit', 'ce', c_epoch_sit, 'sit'),
        'if-ta-ci': Job(0, 'gh', 'both', b_epoch_ci_sit, 'sit', 'ci', c_epoch_sit, 'sit'),
        # sit complex
        # 'complex': Job(0, 'gh', 'both', 0, 'gh', 'ci', 0, 'sit'),
    }
    job = job_table[args.job]

    # main
    if args.mission in ['train', 'full']:
        Mission.train_t2t()
    if args.mission in ['evaluate', 'full']:
        Mission.evaluate_t2t()
    # for review, directly compute scores using stored prediction files
    # the evaluation scores will be shown in the form of a CSV file
    if args.mission == 'review':
        Mission.review_t2t()
