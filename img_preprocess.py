import os
import numpy as np
import imageio

def load_images():
	
	#path = os.getcwd() + '/XRay_data'
	train_images = []
	train_labels = []
	test_images = []
	test_labels = []
	train = ['1','3','4','5','7','9']
	#for reference: test = ['2','6','8','10']	
	#loop through folders
	for i in range(1,11):
		#list the files within the image directories
		str_directory = '/Users/FrancescoMac/Desktop/att_faces_10/s{}'.format(i)
		directory = os.listdir(str_directory)
		for filename in directory:
			#split the filename to get the index
			string = filename.split('.')
			#separate desired training and testing images into separate lists
			if(string[0] in train):
				train_labels.append('s{}'.format(i))
				#read the image
				reshaped = imageio.imread(str_directory+'/'+filename)
				#reshape to reduce the image dimension
				reshaped = reshaped.reshape(10304)
				train_images.append(reshaped)
			else:
				test_labels.append('s{}'.format(i))
				reshaped = imageio.imread(str_directory+'/'+filename)
				reshaped = reshaped.reshape(10304)
				test_images.append(reshaped)

	#convert all lists to numpy arrays		
	train_images = np.asarray(train_images)
	train_labels = np.asarray(train_labels)
	#train_labels = train_labels.reshape((1,60))
	test_images = np.asarray(test_images)
	test_labels = np.asarray(test_labels)
	

	return train_images, train_labels, test_images, test_labels

var1, var2, var3, var4 = load_images()
#save arrays locally
print("now saving files")
np.save("face_train_images", var1)
np.save("face_train_lbls", var2)
np.save("face_test_images", var3)
np.save("face_test_lbls", var4)
import pdb; pdb.set_trace()
print("done saving files")
