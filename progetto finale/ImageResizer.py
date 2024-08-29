import numpy as np
from skimage.transform import resize
from sklearn.base import BaseEstimator, TransformerMixin

class ImageResizer(BaseEstimator, TransformerMixin):
    """
    This transformer resizes input image(s), as a preprocessing step before feature extraction.
    It can be used standalone or integrated inside a standard scikit-learn Pipeline.
    The given image must be grayscale. 
    """
    
    def __init__(self, resize_shape = (128, 64)):
        """
        Class constructor. Its parameter is used as input to scikit-image resize function.

        Parameter:
        - resize_shape: 2-tuple (int, int), default (128, 64)
            Desired final image size, as (height, width).
        """
        self.resize_shape = resize_shape #as row x columns (h x w)
        
    def fit(self, X, y=None):
        """
        This method does nothing, since resizing hasn't a fit process.
        It's included for compatibility with scikit-learn Pipeline.
        """
        return self
    
    def transform(self, X, y=None):
        """
        Resize input image(s) as resize_shape ndarray.

        Parameters:
        - X: (M,N) ndarray or (K,M,N) ndarray.
            If X is a 2d array, it's a single grayscale image;
            If X is a 3d array, it's an array of K grayscale images.
        - y: ignored, included only for compatibility reasons.

        Return:
        resized image as ndarray (2d if X is a single image or 3d if X is an array of images). 
        """
        if X.ndim == 2:  # Single image case
            return resize(X, self.resize_shape).reshape(1, *self.resize_shape)
        return np.array([resize(img, self.resize_shape) for img in X])  
  
