"""
Created on Wed Sep 14 16:21:35 2016
Author: David Malmgren-Hansen
Code for arXiv article: https://arxiv.org/abs/1702.07189

NB! The original results for the article was generated with the DPGMM class in 
scikit-learn ver. 0.17. This class is deprecated in ver. 0.18 and will be 
removed in ver. 0.20. The DPGMM class is replaced by the BayesianGaussianMixture 
with weight_concentration_prior_type="dirichlet_process".
I suspect the underlying intialization scheme of the underlying Gaussian 
mixture components to be slightly different in the BayesianGaussianMixture
class since the new function is senitive to the mean_precision_prior, mean_prior
arguments. The code for a creating similar results with BayesianGaussianMixture
class is therefore left commented out in this file.

Be aware that running the code several times yields slightly different
results due to the random initialization scheme.

Image used in this file is from:
http://www2.warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download
If you intend to use it in research publications please follow their citation 
policy.
"""

import matplotlib.pyplot as plt 
import numpy as np
import keras.backend as K
from keras.applications import vgg16
from sklearn.mixture import DPGMM
#from sklearn.mixture import BayesianGaussianMixture
from scipy.misc import imread, imresize

def get_activations(model, layer, X_batch):
    act_func = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = act_func([X_batch,0])
    return activations

""" Input parameters """
model = vgg16.VGG16()
conf = model.get_config()
layer = 13

""" Load image """
file = "data/warwickQU_train_51.bmp"
if K.image_dim_ordering() == "th":
    imsize = conf['layers'][0]['config']['batch_input_shape'][-1]
else:
    imsize = conf['layers'][0]['config']['batch_input_shape'][-2]
img = imread(file)
img = imresize(img, float(imsize)/np.min(img.shape[:2]), interp='bilinear')[:imsize,:imsize,:]

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original Image')
if K.image_dim_ordering() == "th":
    img = np.moveaxis(img.reshape((1,img.shape[0],img.shape[1],img.shape[2])),-1,1)
img = vgg16.preprocess_input(img.astype('float32'))

""" Scaling activations to fit random initialization scheme"""
actvs = get_activations(model, layer, img).squeeze()
actvs /= np.max(actvs)*0.1

""" Clustering with dirichlet process Gaussian Mixture Model"""
dpgmm = DPGMM(n_components=50, alpha=1, verbose=2, tol=0.01, n_iter=250, min_covar=1e-6)
#dpgmm = BayesianGaussianMixture(n_components=50, covariance_type="diag", reg_covar = 1e-6,
#                                weight_concentration_prior_type="dirichlet_process", 
#                                weight_concentration_prior=1, verbose=2, 
#                                tol=0.01, max_iter=250, init_params='random',
#                                mean_precision_prior=actvs.std(), 
#                                mean_prior=np.repeat(actvs.max()/5,actvs.shape[0]))

dpgmm.fit(np.transpose(actvs.reshape(actvs.shape[0],actvs.shape[1]*actvs.shape[2])))
labels = dpgmm.predict(np.transpose(actvs.reshape(actvs.shape[0],actvs.shape[1]*actvs.shape[2])))
labels = labels.reshape((actvs.shape[1],actvs.shape[2]))

plt.subplot(1,2,2)
plt.imshow(labels, interpolation="nearest")
plt.title('Labelmap from layer '+str(layer))
