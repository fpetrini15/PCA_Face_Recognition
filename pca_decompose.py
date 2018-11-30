import numpy as np
import sklearn
from sklearn import neighbors
from sklearn.decomposition import PCA

#center matrix (training and testing), normalize the training and testing
def pca_decompose():
	#import numpy arrays
	train_images = np.load('/Users/FrancescoMac/Desktop/face_train_images.npy')
	train_labels = np.load('/Users/FrancescoMac/Desktop/face_train_lbls.npy')
	test_images = np.load('/Users/FrancescoMac/Desktop/face_test_images.npy')
	test_labels = np.load('/Users/FrancescoMac/Desktop/face_test_lbls.npy')
	#import pdb; pdb.set_trace()
	#specify the desired rank for decomposition
	ranks = [1,2,3,6,10,20,30]
	for rank in ranks:
		pca = PCA(n_components=rank)
		pca.fit(train_images)
		pca.fit(test_images)
		#apply the dimensionality reduction to the datasets
		train_reduced = pca.transform(train_images)
		test_reduced = pca.transform(test_images)
		
		#import pdb; pdb.set_trace()
		#perform classification using the rank as the no. of neighbors
		knn = neighbors.KNeighborsClassifier()
		k_fit = knn.fit(train_reduced, train_labels)
		print("Accuracy: ",k_fit.score(test_reduced, test_labels))


pca_decompose()



