"""
Sparse Feature Extraction Module
=================================
Constructs:
1. Unigram Bag-of-Words (CountVectorizer)
2. Bigram TF-IDF (TfidfVectorizer)
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import config


class SparseFeatureExtractor:
    """Extracts BoW and TF-IDF sparse feature representations."""

    def __init__(self, min_df=None, max_df=None, bow_max_features=None, tfidf_max_features=None):
        self.min_df = min_df if min_df is not None else config.MIN_DF
        self.max_df = max_df if max_df is not None else config.MAX_DF
        self.bow_max_features = bow_max_features or config.BOW_MAX_FEATURES
        self.tfidf_max_features = tfidf_max_features or config.TFIDF_MAX_FEATURES

        self.bow_vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            max_features=self.bow_max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=config.TFIDF_NGRAM_RANGE,
            max_features=self.tfidf_max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
        )
        self._is_fitted = False

    def fit_transform(self, train_texts):
        """
        Fit vectorizers on training data and return transformed features.

        Parameters
        ----------
        train_texts : list of str
            Preprocessed training texts.

        Returns
        -------
        dict
            {'bow': sparse_matrix, 'tfidf': sparse_matrix}
        """
        n_docs = len(train_texts)
        effective_max_df = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)
        if effective_max_df < self.min_df:
            # Adjust parameters for small subsets so fitting does not fail
            self.bow_vectorizer.set_params(min_df=1, max_df=1.0)
            self.tfidf_vectorizer.set_params(min_df=1, max_df=1.0)
        print("  Fitting Bag-of-Words (unigram)...")
        bow_train = self.bow_vectorizer.fit_transform(train_texts)
        print(f"    BoW vocabulary size: {len(self.bow_vectorizer.vocabulary_):,}")
        print(f"    BoW feature matrix shape: {bow_train.shape}")
        print(f"    BoW sparsity: {1.0 - bow_train.nnz / np.prod(bow_train.shape):.4f}")

        print("  Fitting TF-IDF (bigram)...")
        tfidf_train = self.tfidf_vectorizer.fit_transform(train_texts)
        print(f"    TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_):,}")
        print(f"    TF-IDF feature matrix shape: {tfidf_train.shape}")
        print(f"    TF-IDF sparsity: {1.0 - tfidf_train.nnz / np.prod(tfidf_train.shape):.4f}")

        self._is_fitted = True
        return {'bow': bow_train, 'tfidf': tfidf_train}

    def transform(self, texts):
        """
        Transform new texts using fitted vectorizers.

        Parameters
        ----------
        texts : list of str
            Preprocessed texts to transform.

        Returns
        -------
        dict
            {'bow': sparse_matrix, 'tfidf': sparse_matrix}
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        bow = self.bow_vectorizer.transform(texts)
        tfidf = self.tfidf_vectorizer.transform(texts)
        return {'bow': bow, 'tfidf': tfidf}

    def get_feature_names(self):
        """Return feature names for both vectorizers."""
        return {
            'bow': self.bow_vectorizer.get_feature_names_out(),
            'tfidf': self.tfidf_vectorizer.get_feature_names_out(),
        }
