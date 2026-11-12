"""
Contextual Embedding Extraction Module
========================================
Extracts dense embeddings using:
1. DistilBERT (CLS token + mean-pooled)
2. GloVe (300d mean-pooled)

All embeddings are cached as .npy files for efficiency.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import urllib.request
import zipfile

import config


# ================================================================
# Download Progress Bar Helper
# ================================================================
class _DownloadProgressBar(tqdm):
    """tqdm wrapper for urllib download progress."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# ================================================================
# DistilBERT Embedding Extractor
# ================================================================
class DistilBERTExtractor:
    """Extract CLS and mean-pooled embeddings from DistilBERT."""

    def __init__(self, model_name=None, batch_size=None, max_length=None):
        self.model_name = model_name or config.DISTILBERT_MODEL
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self.max_length = max_length or config.MAX_SEQ_LENGTH

        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  DistilBERT device: {self.device}")

        # Load model and tokenizer
        print(f"  Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print("  [OK] Model loaded successfully")

    def extract_embeddings(self, texts, cache_prefix="distilbert"):
        """
        Extract CLS and mean-pooled embeddings from DistilBERT.

        Parameters
        ----------
        texts : list of str
            Raw (non-preprocessed) texts; BERT has its own tokenizer.
        cache_prefix : str
            Prefix for cache filenames.

        Returns
        -------
        dict
            {'distilbert_cls': ndarray, 'distilbert_mean': ndarray}
        """
        cls_cache = os.path.join(config.CACHE_DIR, f'{cache_prefix}_cls.npy')
        mean_cache = os.path.join(config.CACHE_DIR, f'{cache_prefix}_mean.npy')

        # Check cache
        if os.path.exists(cls_cache) and os.path.exists(mean_cache):
            print(f"  [OK] Loading cached DistilBERT embeddings from {config.CACHE_DIR}")
            cls_embeds = np.load(cls_cache)
            mean_embeds = np.load(mean_cache)
            return {'distilbert_cls': cls_embeds, 'distilbert_mean': mean_embeds}

        print(f"  Extracting DistilBERT embeddings ({len(texts)} texts, batch_size={self.batch_size})...")

        all_cls = []
        all_mean = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="  DistilBERT inference"):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            # Forward pass (no gradient computation)
            with torch.no_grad():
                outputs = self.model(**encoded)

            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            attention_mask = encoded['attention_mask']  # (batch, seq_len)

            # CLS embedding: first token
            cls_embedding = hidden_states[:, 0, :].cpu().numpy()

            # Mean-pooled embedding: average over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embedding = (sum_embeddings / sum_mask).cpu().numpy()

            all_cls.append(cls_embedding)
            all_mean.append(mean_embedding)

        cls_embeds = np.vstack(all_cls)
        mean_embeds = np.vstack(all_mean)

        # Cache embeddings
        np.save(cls_cache, cls_embeds)
        np.save(mean_cache, mean_embeds)
        print(f"  [OK] DistilBERT embeddings cached - CLS: {cls_embeds.shape}, Mean: {mean_embeds.shape}")

        return {'distilbert_cls': cls_embeds, 'distilbert_mean': mean_embeds}


# ================================================================
# GloVe Embedding Extractor
# ================================================================
class GloVeExtractor:
    """Extract mean-pooled GloVe embeddings for documents."""

    def __init__(self, glove_dim=None):
        self.glove_dim = glove_dim or config.GLOVE_DIM
        self.glove_path = os.path.join(config.GLOVE_DIR, config.GLOVE_FILENAME)
        self.embeddings = None

    def _download_glove(self):
        """Download GloVe embeddings if not present."""
        if os.path.exists(self.glove_path):
            return

        zip_path = os.path.join(config.GLOVE_DIR, 'glove.6B.zip')

        if not os.path.exists(zip_path):
            urls = [config.GLOVE_URL, getattr(config, 'GLOVE_BACKUP_URL', 'https://nlp.stanford.edu/data/glove.6B.zip')]
            downloaded = False
            for url in urls:
                print(f"  Downloading GloVe embeddings from: {url}")
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        total_size = int(resp.headers.get('content-length', 0))
                        block_size = 1024 * 1024  # 1MB
                        with open(zip_path, 'wb') as f_out, tqdm(
                            total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="  GloVe download"
                        ) as pbar:
                            while True:
                                chunk = resp.read(block_size)
                                if not chunk:
                                    break
                                f_out.write(chunk)
                                pbar.update(len(chunk))
                    downloaded = True
                    break
                except Exception as e:
                    print(f"  Download from {url} failed ({e}). Trying fallback...")
                    if os.path.exists(zip_path):
                        os.remove(zip_path)

            if not downloaded:
                raise RuntimeError("Could not download GloVe embeddings from any source.")

        print(f"  Extracting {config.GLOVE_FILENAME}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Extract only the file we need
            zf.extract(config.GLOVE_FILENAME, config.GLOVE_DIR)
        print(f"  [OK] GloVe file ready at {self.glove_path}")

    def _load_glove_model(self):
        """Load GloVe vectors into a dictionary."""
        if self.embeddings is not None:
            return

        self._download_glove()

        print(f"  Loading GloVe {self.glove_dim}d vectors into memory...")
        self.embeddings = {}
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="  Loading GloVe", total=400000):
                values = line.strip().split()
                word = values[0]
                try:
                    vector = np.array(values[1:], dtype='float32')
                    if len(vector) == self.glove_dim:
                        self.embeddings[word] = vector
                except ValueError:
                    continue

        print(f"  [OK] Loaded {len(self.embeddings):,} GloVe word vectors")

    def extract_embeddings(self, texts, cache_prefix="glove"):
        """
        Compute mean-pooled GloVe embeddings for each document.

        Parameters
        ----------
        texts : list of str
            Preprocessed texts (lowercased, tokenized).
        cache_prefix : str
            Prefix for cache filename.

        Returns
        -------
        dict
            {'glove': ndarray of shape (n_samples, glove_dim)}
        """
        cache_path = os.path.join(config.CACHE_DIR, f'{cache_prefix}_embeddings.npy')

        if os.path.exists(cache_path):
            print(f"  [OK] Loading cached GloVe embeddings from {cache_path}")
            return {'glove': np.load(cache_path)}

        self._load_glove_model()

        print(f"  Computing GloVe document embeddings ({len(texts)} texts)...")
        doc_embeddings = []
        oov_counts = []

        for text in tqdm(texts, desc="  GloVe embedding"):
            words = text.lower().split()
            word_vectors = []
            oov = 0
            for word in words:
                if word in self.embeddings:
                    word_vectors.append(self.embeddings[word])
                else:
                    oov += 1
            oov_counts.append(oov)

            if word_vectors:
                doc_embedding = np.mean(word_vectors, axis=0)
            else:
                doc_embedding = np.zeros(self.glove_dim, dtype='float32')

            doc_embeddings.append(doc_embedding)

        result = np.vstack(doc_embeddings)

        # Cache
        np.save(cache_path, result)
        avg_oov = np.mean(oov_counts)
        print(f"  [OK] GloVe embeddings shape: {result.shape}")
        print(f"  [OK] Average OOV words per document: {avg_oov:.1f}")

        return {'glove': result}
