import numpy as np
from skimage.transform import resize
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

class ImageResizer(BaseEstimator, TransformerMixin):
    """
    TODO: Write docstring
    """
    
    def __init__(self, resize_shape = (128, 64)):
        self.resize_shape = resize_shape #as row x columns (h x w)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if X.ndim == 2:  # Caso di una singola immagine
            return resize(X, self.output_shape).reshape(1, *self.output_shape)
        return np.array([resize(img, self.output_shape) for img in tqdm(X, desc="Resize image", unit="item")])  
