import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class HOGFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    This transformer computes feature extraction for an image or images array, through HOG method.
    It can be used standalone or integrated inside a standard scikit-learn Pipeline.
    The given image must be grayscale.
    """

    def __init__(self,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Class constructor. Its parameters are used as configruation for hog feature extraction.

        Parameters:
        - orientations: int, default = 9
            Number of orientation bin of HOG histogram.
        - pixels_per_cell: 2-tuple (int, int), default = (8, 8)
            Size (in pixels) of a cell.
        - cells_per_block: 2-tuple (int, int), default = (2, 2)
            Number of cells in each block.

        For more info about HOG feature extraction, visit: 
        https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        """
        This method does nothing, since HOG feature descriptor hasn't a fit process.
        It's included for compatibility with scikit-learn Pipeline.
        """
        return self
    
    def transform(self, X, y=None):
        """
        Perform feature extraction of the input image.

        Parameters:
        - X: (M,N) ndarray or (K,M,N) ndarray.
            If X is a 2d array, it's a single grayscale image;
            If X is a 3 array, it's an array of K grayscale images.
        - y: ignored, included only for compatibility reasons.
                
        Return:
        extracted features matrix as 2d array. 
        """
        if X.ndim == 2:  # Single image case
            return hog(X, pixels_per_cell=self.pixels_per_cell, 
                       cells_per_block=self.cells_per_block, 
                       orientations=self.orientations).reshape(1, -1)
        hog_features = [hog(img, pixels_per_cell=self.pixels_per_cell, 
                            cells_per_block=self.cells_per_block, 
                            # orientations=self.orientations) for img in tqdm(X, desc="HOG feature extraction", unit="item")]
                             orientations=self.orientations) for img in X]
        return np.array(hog_features)
