import os
import random
import toml
from sys import argv
from types import SimpleNamespace
from pathlib import Path

import accelerate
from tqdm import tqdm
import wandb

import dotenv
import numpy as np
import torch
import faiss
from torch.optim.lr_scheduler import LambdaLR
from safetensors import safe_open

from vec2vec.translators.Discriminator import Discriminator

# from eval import eval_model
from vec2vec.utils.collate import MultiencoderTokenizedDataset, TokenizedCollator, identity_collate
from vec2vec.utils.eval_utils import EarlyStopper, eval_loop_
from vec2vec.utils.gan import LeastSquaresGAN, RelativisticGAN, VanillaGAN
from vec2vec.utils.model_utils import get_sentence_embedding_dimension, load_encoder
from vec2vec.utils.utils import *
from vec2vec.utils.streaming_utils import load_streaming_embeddings, process_batch, process_custom_batch
from vec2vec.utils.train_utils import rec_loss_fn, trans_loss_fn, vsp_loss_fn, get_grad_norm
from vec2vec.utils.wandb_logger import Logger

from datasets import load_from_disk

# torch.cuda.set_per_process_memory_fraction(0.5)

gemma_weights = np.load('params.npz')['W_dec']
with safe_open('final.safetensors', framework='pt') as f:
  llama_weights = f.get_tensor('decoder.weight').transpose(0, 1)

dotenv.load_dotenv()
wandb.login(key=os.getenv('WANDB_API_KEY'), force=True)

llama_weights_norm = llama_weights / torch.norm(llama_weights, dim=1, keepdim=True)
llama_weights_norm = llama_weights_norm.to(torch.float32).detach().cpu().numpy()
llama_quantizer = faiss.IndexFlatIP(llama_weights.shape[1])
llama_index = faiss.IndexIVFFlat(llama_quantizer, llama_weights.shape[1], int(llama_weights.shape[0] ** 0.5), 
                                 faiss.METRIC_INNER_PRODUCT)
llama_index.train(llama_weights_norm)
llama_index.add(llama_weights_norm)

gemma_weights_norm = gemma_weights / np.linalg.norm(gemma_weights, axis=1, keepdims=True)
gemma_quantizer = faiss.IndexFlatIP(gemma_weights_norm.shape[1])
gemma_index = faiss.IndexIVFFlat(gemma_quantizer, gemma_weights.shape[1], int(gemma_weights.shape[0] ** 0.5),
                                 faiss.METRIC_INNER_PRODUCT)
gemma_index.train(gemma_weights_norm)
gemma_index.add(gemma_weights_norm)
  
class CustomEmbeddingEncoder:
    def __init__(self, embeddings):
        self.embeddings = torch.tensor(embeddings) if not isinstance(embeddings, torch.Tensor) else embeddings
        self.embedding_dim = self.embeddings.shape[-1]

    def encode(self, indices):
        return self.embeddings[indices]

    def get_sentence_embedding_dimension(self):
        return self.embedding_dim


gemma_enc = CustomEmbeddingEncoder(gemma_weights)
llama_enc = CustomEmbeddingEncoder(llama_weights)

assert gemma_enc.embedding_dim == 3584
assert llama_enc.embedding_dim == 4096


def training_loop_(
    save_dir, accelerator, gan, sup_gan, latent_gan, similarity_gan, translator, sup_dataloader, sup_iter, unsup_dataloader, sup_encs, unsup_enc, cfg, opt, scheduler, logger=None, max_num_batches=None, epoch: int | None = None
):
    device = accelerator.device
    import logging
    if logger is None:
        logger = Logger(dummy=True)

    # wandb.watch(translator, log='all')

    if sup_iter is not None:
        dataloader_pbar = unsup_dataloader
    else:
        dataloader_pbar = zip(sup_dataloader, unsup_dataloader)


    dataloader_pbar = tqdm(dataloader_pbar, total=len(unsup_dataloader), desc="Training")

    model_save_dir = os.path.join(save_dir, 'model.pt')

    translator.train()
    for i, batches in enumerate(dataloader_pbar):
        if sup_iter is not None:
            try:
                sup_batch = next(sup_iter)
            except StopIteration:
                print('Restarting sup_dataloader...')
                sup_iter = iter(sup_dataloader)
                sup_batch = next(sup_iter)
            unsup_batch = batches
        else:
            sup_batch, unsup_batch = batches

        if max_num_batches is not None and i >= max_num_batches:
            print(f"Early stopping at {i} batches")
            break
        with accelerator.accumulate(translator), accelerator.autocast():
            # assert len(set(sup_batch.keys()).intersection(unsup_batch.keys())) == 0
            ins = {
                **process_custom_batch(sup_batch, sup_encs, cfg.normalize_embeddings, device),
                **process_custom_batch(unsup_batch, unsup_enc, cfg.normalize_embeddings, device)
            }

            recons, translations, reps = translator(
                ins, noise_level=cfg.noise_level, include_reps=True
            )
            
            # assess translation quality (cosine similarity search in the other vector space)
            llama_translated = translations['llama']['gemma']
            llama_translated_norm = llama_translated / torch.norm(llama_translated, dim=1, keepdim=True)
            llama_translated_norm = llama_translated_norm.detach().cpu()
            
            llama_sim, _ = llama_index.search(llama_translated_norm, 100)
            llama_avg_cos_sim = np.mean(llama_sim)
            llama_max_cos_sim = np.max(llama_sim)
            
            gemma_translated = translations['gemma']['llama']
            gemma_translated_norm = gemma_translated / torch.norm(gemma_translated, dim=1, keepdim=True)
            gemma_translated_norm = gemma_translated_norm.detach().cpu()
            
            gemma_sim, _ = gemma_index.search(gemma_translated_norm, 100)
            gemma_avg_cos_sim = np.mean(gemma_sim)
            gemma_max_cos_sim = np.max(gemma_sim)

            # discriminator
            disc_r1_penalty, disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc = gan.step(
                real_data=ins[cfg.unsup_emb] + torch.randn_like(ins[cfg.unsup_emb], device=ins[cfg.unsup_emb].device) * cfg.noise_level,
                fake_data=translations[cfg.unsup_emb][cfg.sup_emb] + torch.randn_like(translations[cfg.unsup_emb][cfg.sup_emb], device=translations[cfg.unsup_emb][cfg.sup_emb].device) * cfg.noise_level
            )

            sup_disc_r1_penalty, sup_disc_loss, sup_gen_loss, sup_disc_acc_real, sup_disc_acc_fake, sup_gen_acc = sup_gan.step(
                real_data=ins[cfg.sup_emb] + torch.randn_like(ins[cfg.sup_emb], device=ins[cfg.sup_emb].device) * cfg.noise_level,
                fake_data=translations[cfg.sup_emb][cfg.unsup_emb] + torch.randn_like(translations[cfg.sup_emb][cfg.unsup_emb], device=translations[cfg.sup_emb][cfg.unsup_emb].device) * cfg.noise_level,
            )

            # latent discriminator
            latent_disc_r1_penalty, latent_disc_loss, latent_gen_loss, latent_disc_acc_real, latent_disc_acc_fake, latent_gen_acc = latent_gan.step(
                real_data=reps[cfg.sup_emb],
                fake_data=reps[cfg.unsup_emb]
            )

            # similarity discriminator
            if cfg.loss_coefficient_similarity_gen > 0:
                real_sims_A = ins[cfg.sup_emb] @ ins[cfg.sup_emb].T
                fake_sims_A = (
                    translations[cfg.sup_emb][cfg.unsup_emb] @ translations[cfg.sup_emb][cfg.unsup_emb].T
                )
                real_sims_B = ins[cfg.unsup_emb] @ ins[cfg.unsup_emb].T
                fake_sims_B = (
                    translations[cfg.unsup_emb][cfg.sup_emb] @ translations[cfg.unsup_emb][cfg.sup_emb].T
                )
                similarity_r1_penalty, similarity_disc_loss, similarity_gen_loss, similarity_disc_acc_real, similarity_disc_acc_fake, similarity_gen_acc = similarity_gan.step(
                    real_data=torch.cat([real_sims_A, real_sims_B], dim=1),
                    fake_data=torch.cat([fake_sims_A, fake_sims_B], dim=1)
                )
            else:
                similarity_r1_penalty = torch.tensor(0.0)
                similarity_disc_loss = torch.tensor(0.0)
                similarity_gen_loss = torch.tensor(0.0)
                similarity_disc_acc_real = 0.0
                similarity_disc_acc_fake = 0.0
                similarity_gen_acc = 0.0

            rec_loss = rec_loss_fn(ins, recons, logger)
            ins_reversed = {
                cfg.sup_emb: ins[cfg.unsup_emb],
                cfg.unsup_emb: ins[cfg.sup_emb],
            }
            translations_as_recons = {
                cfg.sup_emb: translations[cfg.unsup_emb][cfg.sup_emb],
                cfg.unsup_emb: translations[cfg.sup_emb][cfg.unsup_emb],
            }
            reverse_rec_loss = rec_loss_fn(ins_reversed, translations_as_recons, logger, prefix="reverse_")

            recons_as_translations = {
                in_name: { in_name: val } for in_name, val in recons.items()
            }
            vsp_loss = vsp_loss_fn(ins, recons_as_translations, logger)
            if (cfg.loss_coefficient_cc_rec > 0) or (cfg.loss_coefficient_cc_trans > 0):
                cc_ins = {}
                for out_flag in translations.keys():
                    in_flag = random.choice(list(translations[out_flag].keys()))
                    cc_ins[out_flag] = translations[out_flag][in_flag].detach()
                cc_recons, cc_translations = translator(cc_ins)
                cc_rec_loss = rec_loss_fn(ins, cc_recons, logger, prefix="cc_")
                cc_trans_loss = trans_loss_fn(ins, cc_translations, logger, prefix="cc_")
                cc_vsp_loss = vsp_loss_fn(ins, cc_translations, logger)
            else:
                cc_rec_loss = torch.tensor(0.0)
                cc_trans_loss = torch.tensor(0.0)
                cc_vsp_loss = torch.tensor(0.0)

            loss = (
                + (rec_loss * cfg.loss_coefficient_rec)
                + (reverse_rec_loss * cfg.loss_coefficient_reverse_rec)
                + (vsp_loss * cfg.loss_coefficient_vsp)
                + (cc_vsp_loss * cfg.loss_coefficient_cc_vsp)
                + (cc_rec_loss * cfg.loss_coefficient_cc_rec)
                + (cc_trans_loss * cfg.loss_coefficient_cc_trans)
                + (gen_loss * cfg.loss_coefficient_gen)
                + (sup_gen_loss * cfg.loss_coefficient_gen)
                + (latent_gen_loss * cfg.loss_coefficient_latent_gen)
                + (similarity_gen_loss * cfg.loss_coefficient_similarity_gen)
            )
            exit_on_nan(loss)
            opt.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(translator.parameters(), cfg.max_grad_norm)
            grad_norm_generator = get_grad_norm(translator)
            grad_norm_discriminator = get_grad_norm(gan.discriminator)
            grad_norm_sup_discriminator = get_grad_norm(sup_gan.discriminator)
            grad_norm_latent_discriminator = get_grad_norm(latent_gan.discriminator)
            grad_norm_similarity_discriminator = get_grad_norm(similarity_gan.discriminator)

            opt.step()
            scheduler.step()

            metrics = {
                "disc_loss": disc_loss.item(),
                "disc_r1_penalty": disc_r1_penalty.item(),
                "sup_disc_loss": sup_disc_loss.item(),
                "sup_disc_r1_penalty": sup_disc_r1_penalty.item(),
                "latent_disc_loss": latent_disc_loss.item(),
                "latent_disc_r1_penalty": latent_disc_r1_penalty.item(),
                "similarity_disc_loss": similarity_disc_loss.item(),
                "similarity_r1_penalty": similarity_r1_penalty.item(),
                "rec_loss": rec_loss.item(),
                "reverse_rec_loss": reverse_rec_loss.item(),
                "vsp_loss": vsp_loss.item(),
                "cc_vsp_loss": cc_vsp_loss.item(),
                "cc_rec_loss": cc_rec_loss.item(),
                "cc_trans_loss": cc_trans_loss.item(),
                "gen_loss": gen_loss.item(),
                "sup_gen_loss": sup_gen_loss.item(),
                "latent_gen_loss": latent_gen_loss.item(),
                "similarity_gen_loss": similarity_gen_loss.item(),
                "loss": loss.item(),
                "grad_norm_generator": grad_norm_generator,
                "grad_norm_discriminator": grad_norm_discriminator,
                "grad_norm_sup_discriminator": grad_norm_sup_discriminator,
                "grad_norm_latent_discriminator": grad_norm_latent_discriminator,
                "grad_norm_similarity_discriminator": grad_norm_similarity_discriminator,
                "learning_rate": opt.param_groups[0]["lr"],
                "disc_acc_real": disc_acc_real,
                "disc_acc_fake": disc_acc_fake,
                "latent_disc_acc_real": latent_disc_acc_real,
                "latent_disc_acc_fake": latent_disc_acc_fake,
                "gen_acc": gen_acc,
                "sup_disc_acc_real": sup_disc_acc_real,
                "sup_disc_acc_fake": sup_disc_acc_fake,
                "sup_gen_acc": sup_gen_acc,
                "similarity_disc_acc_real": similarity_disc_acc_real,
                "similarity_disc_acc_fake": similarity_disc_acc_fake,
                "similarity_gen_acc": similarity_gen_acc,
                "gemma2llama_avg_cos_sim_train": llama_avg_cos_sim,
                "gemma2llama_max_cos_sim_train": llama_max_cos_sim,
                "llama2gemma_avg_cos_sim_train": gemma_avg_cos_sim,
                "llama2gemma_max_cos_sim_train": gemma_max_cos_sim
            }

            for metric, value in metrics.items():
                logger.logkv(metric, value)
            logger.dumpkvs(force=(hasattr(cfg, 'force_dump') and cfg.force_dump))
            dataloader_pbar.set_postfix(metrics)

    with open(save_dir + 'config.toml', 'w') as f:
        toml.dump(cfg.__dict__, f)
    if (epoch + 1) % 25 == 0:
        torch.save(accelerator.unwrap_model(translator).state_dict(), Path(save_dir) / f'epoch_{epoch}.pt')
    return sup_iter

class CustomEmbeddingDataset:
    def __init__(self, embedding_indices, batch_size, n_embs_per_batch=1, seed=42):
        self.indices = embedding_indices
        self.batch_size = batch_size
        self.n_embs_per_batch = n_embs_per_batch
        self.seed = seed
        self._prepare_batches()

    def _prepare_batches(self):
        np.random.seed(self.seed)
        shuffled_indices = np.random.permutation(self.indices)

        self.batches = []
        for i in range(0, len(shuffled_indices), self.batch_size):
            batch_indices = shuffled_indices[i:i+self.batch_size]
            if len(batch_indices) == self.batch_size:
                self.batches.append(batch_indices)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def __iter__(self):
        for batch in self.batches:
            yield batch

def create_custom_datasets(cfg):
    total_embeddings = min(gemma_weights.shape[0], llama_weights.shape[0])
    all_indices = np.arange(total_embeddings)

    np.random.seed(cfg.val_dataset_seed)
    val_indices = np.random.choice(all_indices, size=cfg.val_size, replace=False)
    train_indices = np.setdiff1d(all_indices, val_indices)

    np.random.seed(cfg.train_dataset_seed)
    train_indices = np.random.permutation(train_indices)

    if hasattr(cfg, 'num_points'):
        sup_indices = train_indices[:cfg.num_points]
        unsup_indices = train_indices[:cfg.num_points]
    elif hasattr(cfg, 'unsup_points'):
        unsup_indices = train_indices[:cfg.unsup_points]
        sup_indices = train_indices[cfg.unsup_points:]

    supset = CustomEmbeddingDataset(sup_indices, cfg.bs, cfg.n_embs_per_batch, cfg.sampling_seed)
    unsupset = CustomEmbeddingDataset(unsup_indices, cfg.bs, 1, cfg.sampling_seed)
    valset = CustomEmbeddingDataset(val_indices, cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs, 2, cfg.sampling_seed)

    return supset, unsupset, valset

def main(experiment: str = 'unsupervised'):
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'configs/{experiment}.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**{k: v for d in cfg.values() for k, v in d.items()}, **unknown_cfg})

    if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision != 'no' and cfg.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        cfg.mixed_precision = 'fp16'
        cfg.gradient_accumulation_steps = 1
        print("Note: bf16 is not available on this hardware! Reverting to fp16 and setting accumulation steps to 1.")

    # set seeds
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # use_val_set = hasattr(cfg, 'val_size')
    use_val_set = False

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision != 'no' else None,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )
    # https://github.com/huggingface/transformers/issues/26548
    accelerator.dataloader_config.dispatch_batches = False

    if hasattr(cfg, 'force_wandb_name') and cfg.force_wandb_name:
        save_dir = cfg.save_dir.format(cfg.wandb_name)
    else:
        cfg.wandb_name = ','.join([f"{k[0]}:{v}" for k, v in unknown_cfg.items()]) if unknown_cfg else cfg.wandb_name
        save_dir = cfg.save_dir.format(cfg.latent_dims if hasattr(cfg, 'latent_dims') else cfg.wandb_name)

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=(cfg.wandb_project is None) or not (cfg.use_wandb),
        config=cfg,
    )

    print("Running Experiment:", cfg.wandb_name)


    # sup_encs = {
    #     cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)
    # }
    # encoder_dims = {
    #     cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])
    # }
    sup_encs = {
        cfg.sup_emb: gemma_enc
    }
    encoder_dims = {
        cfg.sup_emb: gemma_enc.embedding_dim
    }
    translator = load_n_translator(cfg, encoder_dims)

    model_save_dir = os.path.join(save_dir, 'model.pt')
    disc_save_dir = os.path.join(save_dir, 'disc.pt')

    os.makedirs(save_dir, exist_ok=True)

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {
        cfg.unsup_emb: llama_enc
    }
    unsup_dim = {
        cfg.unsup_emb: llama_enc.embedding_dim
    }
    translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])

    assert cfg.unsup_emb not in sup_encs
    assert cfg.unsup_emb in translator.in_adapters
    assert cfg.unsup_emb in translator.out_adapters

    cfg.num_params = sum(x.numel() for x in translator.parameters())
    print("Number of parameters:", cfg.num_params)
    print("Number of *trainable* parameters:", sum(p.numel() for p in translator.parameters() if p.requires_grad))
    print(translator)

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=(cfg.wandb_project is None) or not (cfg.use_wandb),
        config=cfg,
    )

    num_workers = min(get_num_proc(), 8)
    # if cfg.dataset != 'mimic':
    #     dset = load_streaming_embeddings(cfg.dataset)
    #     print(f"Using {num_workers} workers and {len(dset)} datapoints")
# 
    #     dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    #     dset = dset_dict["train"]
    #     valset = dset_dict["test"]
# 
    #     assert hasattr(cfg, 'num_points') or hasattr(cfg, 'unsup_points')
    #     dset = dset.shuffle(seed=cfg.train_dataset_seed)
    #     if hasattr(cfg, 'num_points'):
    #         assert cfg.num_points > 0 and cfg.num_points <= len(dset) // 2
    #         supset = dset.select(range(cfg.num_points))
    #         unsupset = dset.select(range(cfg.num_points, cfg.num_points * 2))
    #     elif hasattr(cfg, 'unsup_points'):
    #         unsupset = dset.select(range(min(cfg.unsup_points, len(dset))))
    #         supset = dset.select(range(min(cfg.unsup_points, len(dset)), len(dset) - len(unsupset)))
    # else:
    #    supset = load_from_disk('data/mimic')['supervised'].shuffle(cfg.train_dataset_seed).select(range(cfg.num_points))
    #    unsupset = load_from_disk('data/mimic')['unsupervised'].shuffle(cfg.train_dataset_seed).select(range(cfg.num_points))
    #    valset = load_from_disk('data/mimic')['evaluation'].shuffle(cfg.val_dataset_seed).select(range(cfg.val_size))
#
    #    # for each, drop all columns but 'text' using remove_columns
    #    supset = supset.remove_columns([col for col in supset.column_names if col != 'text'])
    #    unsupset = unsupset.remove_columns([col for col in unsupset.column_names if col != 'text'])
    #    valset = valset.remove_columns([col for col in valset.column_names if col != 'text'])


    supset, unsupset, _ = create_custom_datasets(cfg)

    sup_dataloader = DataLoader(
        supset,
        batch_size=1,
        num_workers=num_workers // 2,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=None,
        collate_fn=identity_collate,
        drop_last=True,
    )
    unsup_dataloader = DataLoader(
        unsupset,
        batch_size=1,
        num_workers=num_workers // 2,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=None,
        collate_fn=identity_collate,
        drop_last=True,
    )
    
    _, _, valset = create_custom_datasets(cfg)
    if use_val_set:
        valloader = DataLoader(
            valset,
            batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=(8 if num_workers > 0 else None),
            collate_fn=identity_collate,
            drop_last=True,
        )
        valloader = accelerator.prepare(valloader)

    opt = torch.optim.Adam(translator.parameters(), lr=cfg.lr, fused=False, betas=(0.5, 0.999))

    print(f'{translator.in_adapters = }')
    print(f'{translator.out_adapters = }')

    ######################################################################################
    disc = Discriminator(
        latent_dim=translator.in_adapters[cfg.unsup_emb].in_dim,
        discriminator_dim=cfg.disc_dim,
        depth=cfg.disc_depth,
        weight_init=cfg.weight_init
    )
    disc_opt = torch.optim.Adam(disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps, betas=(0.5, 0.999))

    cfg.num_disc_params = sum(x.numel() for x in disc.parameters())
    print(f"Number of discriminator parameters:", cfg.num_disc_params)
    ######################################################################################
    sup_disc = Discriminator(
        latent_dim=translator.in_adapters[cfg.sup_emb].in_dim,
        discriminator_dim=cfg.disc_dim,
        depth=cfg.disc_depth,
    )
    sup_disc_opt = torch.optim.Adam(sup_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps, betas=(0.5, 0.999))

    cfg.num_sup_disc_params = sum(x.numel() for x in sup_disc.parameters())
    print(f"Number of supervised discriminator parameters:", cfg.num_sup_disc_params)
    print(sup_disc)
    ######################################################################################
    latent_disc = Discriminator(
        latent_dim=cfg.d_adapter,
        discriminator_dim=cfg.disc_dim,
        depth=cfg.disc_depth,
        weight_init=cfg.weight_init
    )
    latent_disc_opt = torch.optim.RMSprop(latent_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps)
    cfg.num_latent_disc_params = sum(x.numel() for x in latent_disc.parameters())
    print(f"Number of latent discriminator parameters:", cfg.num_latent_disc_params)
    print(latent_disc)
    latent_disc_opt = torch.optim.Adam(latent_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps, betas=(0.5, 0.999))
    ######################################################################################
    similarity_disc = Discriminator(
        latent_dim=cfg.bs,
        discriminator_dim=cfg.disc_dim,
        depth=cfg.disc_depth,
        weight_init=cfg.weight_init
    )
    similarity_disc_opt = torch.optim.RMSprop(similarity_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps)
    cfg.num_similarity_disc_params = sum(x.numel() for x in similarity_disc.parameters())
    print(f"Number of similarity discriminator parameters:", cfg.num_similarity_disc_params)
    print(similarity_disc)
    similarity_disc_opt = torch.optim.Adam(similarity_disc.parameters(), lr=cfg.disc_lr, eps=cfg.eps, betas=(0.5, 0.999))
    ######################################################################################

    max_num_epochs = int(np.ceil(cfg.epochs))
    steps_per_epoch = len(supset) // cfg.bs
    total_steps = steps_per_epoch * cfg.epochs / cfg.gradient_accumulation_steps
    warmup_length = (cfg.warmup_length if hasattr(cfg, 'warmup_length') else 100)

    def lr_lambda(step):
        if step < warmup_length:
            return min(1, step / warmup_length)
        else:
            if hasattr(cfg, 'no_scheduler') and cfg.no_scheduler:
                return 1
            return 1 - (step - warmup_length) / max(1, total_steps - warmup_length)

    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)
    disc_scheduler = LambdaLR(disc_opt, lr_lambda=lr_lambda)
    sup_disc_scheduler = LambdaLR(sup_disc_opt, lr_lambda=lr_lambda)
    latent_disc_scheduler = LambdaLR(latent_disc_opt, lr_lambda=lr_lambda)
    similarity_disc_scheduler = LambdaLR(similarity_disc_opt, lr_lambda=lr_lambda)

    if cfg.finetune_mode:
        assert hasattr(cfg, 'load_dir')
        print(f"Loading models from {cfg.load_dir}...")
        translator.load_state_dict(torch.load(cfg.load_dir + 'model.pt', map_location='cpu'), strict=False)
        disc.load_state_dict(torch.load(cfg.load_dir + 'disc.pt', map_location='cpu'))

    translator, opt, scheduler = accelerator.prepare(translator, opt, scheduler)
    disc, disc_opt, disc_scheduler = accelerator.prepare(disc, disc_opt, disc_scheduler)
    sup_disc, sup_disc_opt, sup_disc_scheduler = accelerator.prepare(sup_disc, sup_disc_opt, sup_disc_scheduler)
    latent_disc, latent_disc_opt, latent_disc_scheduler = accelerator.prepare(latent_disc, latent_disc_opt, latent_disc_scheduler)
    similarity_disc, similarity_disc_opt, similarity_disc_scheduler = accelerator.prepare(
        similarity_disc, similarity_disc_opt, similarity_disc_scheduler
    )
    sup_dataloader, unsup_dataloader = accelerator.prepare(sup_dataloader, unsup_dataloader)


    if cfg.gan_style == "vanilla":
        gan_cls = VanillaGAN
    elif cfg.gan_style == "least_squares":
        gan_cls = LeastSquaresGAN
    elif cfg.gan_style == "relativistic":
        gan_cls = RelativisticGAN
    else:
        raise ValueError(f"Unknown GAN style: {cfg.gan_style}")
    latent_gan = gan_cls(
        cfg=cfg,
        generator=translator,
        discriminator=latent_disc,
        discriminator_opt=latent_disc_opt,
        discriminator_scheduler=latent_disc_scheduler,
        accelerator=accelerator,
    )
    similarity_gan = gan_cls(
        cfg=cfg,
        generator=translator,
        discriminator=similarity_disc,
        discriminator_opt=similarity_disc_opt,
        discriminator_scheduler=similarity_disc_scheduler,
        accelerator=accelerator,
    )
    gan = gan_cls(
        cfg=cfg,
        generator=translator,
        discriminator=disc,
        discriminator_opt=disc_opt,
        discriminator_scheduler=disc_scheduler,
        accelerator=accelerator,
    )
    sup_gan = gan_cls(
        cfg=cfg,
        generator=translator,
        discriminator=sup_disc,
        discriminator_opt=sup_disc_opt,
        discriminator_scheduler=sup_disc_scheduler,
        accelerator=accelerator
    )

    sup_iter = None
    if hasattr(cfg, 'unsup_points'):
        sup_iter = iter(sup_dataloader)

    if hasattr(cfg, 'val_size') and hasattr(cfg, 'patience') and hasattr(cfg, 'min_delta'):
        early_stopper = EarlyStopper(patience=cfg.patience, min_delta=cfg.min_delta, increase=False)
        early_stopping = True
    else:
        early_stopping = False

    for epoch in range(max_num_epochs):
        if use_val_set:
            with torch.no_grad(), accelerator.autocast():
                translator.eval()
                val_res = {}
                recons, trans, heatmap_dict, _, _, _ = eval_loop_(cfg, translator, {**sup_encs, **unsup_enc}, valloader, device=accelerator.device)
                for flag, res in recons.items():
                    for k, v in res.items():
                        if k == 'cos':
                            val_res[f"val/rec_{flag}_{k}"] = v
                for target_flag, d in trans.items():
                    for flag, res in d.items():
                        for k, v in res.items():
                            if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                                continue
                            val_res[f"val/{flag}_{target_flag}_{k}"] = v

                if len(heatmap_dict) > 0:
                    for k,v in heatmap_dict.items():
                        if "heatmap" in k and 'top' not in k:
                            v = wandb.Image(v)
                            val_res[f"val/{k}"] = v
                        else:
                            val_res[f"val/{k} (avg. {cfg.top_k_batches} batches)"] = v
                wandb.log(val_res)
                translator.train()

            if epoch >= cfg.min_epochs and early_stopping:
                score = np.mean([v for k, v in val_res.items() if 'top_rank' in k])

                if early_stopper.early_stop(score):
                    print("Early stopping...")
                    break
                if early_stopper.counter == 0 and score < early_stopper.opt_val:
                    print(f"Saving model (counter = {early_stopper.counter})... {score} < {early_stopper.opt_val} is the best score so far...")
                    save_everything(cfg, translator, opt, [gan, sup_gan, latent_gan, similarity_gan], save_dir)

        max_num_batches = None
        print(f"Epoch", epoch, "max_num_batches", max_num_batches, "max_num_epochs", max_num_epochs)
        if epoch + 1 >= max_num_epochs:
            max_num_batches = max(1, (cfg.epochs - epoch) * len(supset) // cfg.bs)
            print(f"Setting max_num_batches to {max_num_batches}")

        sup_iter = training_loop_(
            save_dir=save_dir,
            accelerator=accelerator,
            translator=translator,
            gan=gan,
            sup_gan=sup_gan,
            latent_gan=latent_gan,
            similarity_gan=similarity_gan,
            sup_dataloader=sup_dataloader,
            sup_iter=sup_iter,
            unsup_dataloader=unsup_dataloader,
            sup_encs=sup_encs,
            unsup_enc=unsup_enc,
            cfg=cfg,
            opt=opt,
            scheduler=scheduler,
            logger=logger,
            max_num_batches=max_num_batches,
            epoch=epoch
        )

    with open(save_dir + 'config.toml', 'w') as f:
        toml.dump(cfg.__dict__, f)

if __name__ == '__main__':
    os.chdir('vec2vec')
    
    main()

