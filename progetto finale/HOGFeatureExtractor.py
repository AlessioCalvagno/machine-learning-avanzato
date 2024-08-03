import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class HOGFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    TODO: write docstring
    """

    def __init__(self,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if X.ndim == 3:  # Caso di una singola immagine
            return hog(X, pixels_per_cell=self.pixels_per_cell, 
                       cells_per_block=self.cells_per_block, 
                       orientations=self.orientations).reshape(1, -1)
        hog_features = [hog(img, pixels_per_cell=self.pixels_per_cell, 
                            cells_per_block=self.cells_per_block, 
                            orientations=self.orientations) for img in tqdm(X, desc="HOG feature extraction", unit="item")]
        return np.array(hog_features)
