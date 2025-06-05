import torch
import numpy as np
import toml
import faiss

from types import SimpleNamespace
from safetensors import safe_open

from vec2vec.utils.utils import load_n_translator
from vec2vec.translators.TransformTranslator import TransformTranslator

gemma_weights = np.load('params.npz')['W_dec']
with safe_open('final.safetensors', framework='pt') as f:
  llama_weights = f.get_tensor('decoder.weight').transpose(0, 1)
  
print(f'{gemma_weights.shape = }')
print(f'{llama_weights.shape = }')

def load_pretrained_translator(model_path: str):
  cfg = toml.load('vec2vec/configs/unsupervised.toml')
  cfg = SimpleNamespace(**{k: v for d in cfg.values() for k, v in d.items()})
  
  # weird architecture from original code
  encoder_dims = {cfg.sup_emb: 3584}
  translator = load_n_translator(cfg, encoder_dims)
  
  unsup_dim = {cfg.unsup_emb: 4096}
  translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])
  
  state_dict = torch.load(model_path, map_location='cpu')
  translator.load_state_dict(state_dict, strict=False)
  
  return translator, cfg
  

def inference(translator: TransformTranslator, embeddings: np.ndarray | torch.Tensor, from_: str, to_: str, device='cpu'):
  translator.eval().to(device)
  
  if isinstance(embeddings, np.ndarray):
    embeddings = torch.from_numpy(embeddings).float()
  embeddings.to(device)
  
  inputs = {from_: embeddings.unsqueeze(0)}
  with torch.no_grad():
    _, translations = translator(inputs, out_set = set([to_]))
    return translations[to_][from_]
    
if __name__ == '__main__':
  output_space_neighbors = True
  
  translator, _ = load_pretrained_translator('checkpoint_0506_1126/epoch_475.pt')
  
  llama_weights = llama_weights.to(torch.float32).detach().cpu().numpy()
  
  # llama_weights_norm = llama_weights / torch.norm(llama_weights, dim=1, keepdim=True)
  # llama_weights_norm = llama_weights_norm.to(torch.float32).detach().cpu().numpy()
  # llama_quantizer = faiss.IndexFlatIP(llama_weights.shape[1])
  # llama_index = faiss.IndexIVFFlat(llama_quantizer, llama_weights.shape[1], int(llama_weights.shape[0] ** 0.5), 
  #                                  faiss.METRIC_INNER_PRODUCT)
  # llama_index.train(llama_weights_norm)
  # llama_index.add(llama_weights_norm)

  gemma_weights_norm = gemma_weights / np.linalg.norm(gemma_weights, axis=1, keepdims=True)
  gemma_quantizer = faiss.IndexFlatIP(gemma_weights_norm.shape[1])
  gemma_index = faiss.IndexIVFFlat(gemma_quantizer, gemma_weights.shape[1], int(gemma_weights.shape[0] ** 0.5),
                                   faiss.METRIC_INNER_PRODUCT)
  gemma_index.train(gemma_weights_norm)
  gemma_index.add(gemma_weights_norm)
  
  min_seed = 100_000
  max_seed = 200_000
  
  with open(f'latest_inference_run_{min_seed}:{max_seed}.log', 'w', encoding='utf-8') as logfile:
    for seed in range(min_seed, max_seed):
      np.random.seed(seed)

      eval_indices = np.random.choice(np.arange(len(llama_weights)), size=1)
      eval_features = llama_weights[eval_indices]

      for idx, feature in zip(eval_indices, eval_features):
        gemma_translated = inference(translator, feature, from_='llama', to_='gemma').squeeze()
        gemma_translated_norm = gemma_translated / np.linalg.norm(gemma_translated)

        assert len(feature) == llama_weights.shape[1]
        assert len(gemma_translated) == gemma_weights.shape[1]

        similarities, indices = gemma_index.search(np.expand_dims(gemma_translated_norm, 0), 100)

        closest = np.max(similarities[0])
        furthest = np.min(similarities[0])

        print(f'top100 cosine sim. for {seed = } range: [{furthest:.3f}; {closest:.3f}]', file=logfile)

        close_enough = similarities[0] > 0.7
        neighbors = indices[0][close_enough]
        if len(neighbors) != 0:
          idx = int(idx)
          print(f'{seed = }, {idx = }, {neighbors = }', file=logfile)
          
          if output_space_neighbors:
            space_neighbors_sim, space_neighbors_ind = gemma_index.search(np.expand_dims(neighbors[0], 0), 100)
            print(f'{space_neighbors_sim[:10] = }')
            print(f'{space_neighbors_ind[:10] = }')

        print('-' * 15, file=logfile)
