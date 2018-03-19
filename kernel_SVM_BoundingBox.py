# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:11:39 2018

@author: alex

Implement RBF kernel SVM with Bounding Box and Parameter Tuning
"""

import numpy  as np  
import cv2
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
x = np.loadtxt("train_x.csv", delimiter=",") # load from text 
y = np.loadtxt("train_y.csv", delimiter=",")

TRAIN_X = x.reshape(-1, 64, 64)
TRAIN_Y = y.reshape(-1)


# Find bounding boxes of all digits (Order from largest to smallest)
def get_bounding_boxes(x):
    BB = []
    for i in range(x.shape[0]):
        img = x[i].copy()

        # Threshold, find contours (connected parts of image with the same value)
        ret_t, thresh = cv2.threshold(img,254,255,0)
        im2, contours, hierarchy = cv2.findContours(np.uint8(thresh), 0, 2)

        # Find min area rectangle that will encompass the contour
        L = []
        for c in contours:
            rect = cv2.minAreaRect(c)
            pos, size, orient = rect
            area = max(size[0],size[1])**2
            # Discard bounding boxes that cannot possibly be a digit..
            if area > 49:
                L.append((area,[pos[0],pos[1],size[0],size[1],orient]))
        
        # Sort by area, from largest to smallest
        L.sort(key=lambda x: x[0], reverse=True)
        L = list(list(zip(*L))[1])
        BB.append(L)
    
    return BB


# Return coordinates of a tight, non-oriented box around each digit
def get_coord(img, bounding_box, offset=0):
    _, img_t = cv2.threshold(img,254,255,0)
    x_pos, y_pos, width, height, orient = bounding_box
    box = cv2.boxPoints(((x_pos, y_pos),(width, height),orient))
    box = np.int0(box)

    x_min = max(min(box[:,0]),0)
    x_max = min(max(box[:,0]),img_t.shape[0])
    y_min = max(min(box[:,1]),0)
    y_max = min(max(box[:,1]),img_t.shape[0])
    
    # Original bounding box found (discarding orientation...)
    digit = img_t[y_min:y_max,x_min:x_max].copy()
    
    # Since we've discarded orientation info, we should tighten up the bounding box to compensate
    s_x = np.sum(digit,axis=0)
    x_nz = np.nonzero(s_x)
    x_min += np.amin(x_nz)
    x_max -= (digit.shape[1] - np.amax(x_nz))

    s_y = np.sum(digit,axis=1)
    y_nz = np.nonzero(s_y)
    y_min += np.amin(y_nz)
    y_max -= (digit.shape[0] - np.amax(y_nz))
    
    x_min = max(x_min-offset,0)
    x_max = min(x_max+offset,img_t.shape[0])
    y_min = max(y_min-offset,0)
    y_max = min(y_max+offset,img_t.shape[0])
    
    width = x_max-x_min
    height = y_max-y_min
    
    return x_min, y_min, width, height


# Simply indexes into the image and returns the part of the image corresponding to the non-oriented bounded box around the rectangle 
def get_digit(img, bounding_box, offset=0):
    x, y, w, h = get_coord(img, bounding_box)
    digit = digit[y:y+h,x:x+w].copy()
    return digit



# Oriented bounding boxes for all digits
TRAIN_BB = get_bounding_boxes(TRAIN_X)

DISCARD_THRESH = 34
TRAIN_X_ORIG = []
TRAIN_X_PROC = []
TRAIN_Y_PROC = []
TRAIN_BB_PROC = []
for i in range(TRAIN_X.shape[0]):
    
    # Cannot possibly be a single digit... Discard...
    x_orig, y_orig, w_orig, h_orig, _ = TRAIN_BB[i][0]
    if w_orig >= DISCARD_THRESH or h_orig >= DISCARD_THRESH:
      continue
    
    # Get un-rotated digit and threshold...
    x,y,w,h = get_coord(TRAIN_X[i], TRAIN_BB[i][0])
    x,y,w,h = int(x),int(y),int(w),int(h)
    _, img_t = cv2.threshold(TRAIN_X[i,y:y+h,x:x+w],254,255,0)  
    
    # Pad such that we have a square...
    max_wh = max(w,h)
    if max_wh > w:
      pad_amt = int((max_wh-w)/2)
      img_t = np.pad(img_t, ((0,0),(pad_amt,pad_amt)), 'constant', constant_values=0)
    elif max_wh > h:
      pad_amt = int((max_wh-h)/2)
      img_t = np.pad(img_t, ((pad_amt,pad_amt),(0,0)), 'constant', constant_values=0)
    
    # Pad such that we have a border of 2 pixels...
    img_t = np.pad(img_t, 2, 'constant', constant_values=0)
    img_t = cv2.resize(img_t, (28,28))
    
    TRAIN_X_ORIG.append(TRAIN_X[i])
    TRAIN_X_PROC.append(img_t)
    TRAIN_Y_PROC.append(TRAIN_Y[i])
    TRAIN_BB_PROC.append(TRAIN_BB[i])
    
TRAIN_X_ORIG = np.array(TRAIN_X_ORIG)
TRAIN_X_PROC = np.expand_dims(np.array(TRAIN_X_PROC),axis=1)
TRAIN_Y_PROC = np.array(TRAIN_Y_PROC)
TRAIN_BB_PROC = np.array(TRAIN_BB_PROC)

# Normalize to be in range [-1,1]
TRAIN_X_PROC = (TRAIN_X_PROC/255)*2-1


#reshape bouding box to 1d vector
xtrain=TRAIN_X_PROC.reshape(49527,-1)
ytrain=TRAIN_Y_PROC

#train valid split
X_train, X_test, y_train, y_test = train_test_split(xtrain[:10000,:], ytrain[:10000], test_size=0.2, 
                                                    random_state=1)


#parameter tuning on validation set

#range1
range1=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
score1=np.zeros(len(range1))

for i in range(len(range1)):
    clf1 = SVC(gamma=range1[i])
    clf1.fit(X_train, y_train)
    score1[i]=clf1.score(X_test,y_test)


plt.figure()
plt.plot(np.log10(range1),score1)
plt.title('RBF kernel SVM')
plt.xlabel('gamma (log)')
plt.ylabel('Accuracy')
plt.grid(color='b' , linewidth='0.1' ,linestyle = "-.")
plt.legend(loc='best')
plt.show()

#range2
range2=np.linspace(1e-3,1e-1,20)
score2=np.zeros(len(range2))


for j in range(len(range2)):
    clf1 = SVC(gamma=range2[j])
    clf1.fit(X_train, y_train)
    score2[j]=clf1.score(X_test,y_test)



plt.plot(range2,score2)
plt.title('RBF kernel SVM')
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.grid(color='b' , linewidth='0.1' ,linestyle = "-.")
plt.legend(loc='best')
plt.show()


#range3
range3=np.linspace(0.01,0.15,10)
score3=np.zeros(len(range3))

for j in range(len(range3)):
    clf1 = SVC(gamma=range3[j])
    clf1.fit(X_train, y_train)
    score3[j]=clf1.score(X_test,y_test)


plt.figure()
plt.plot(range3,score3)
plt.title('RBF kernel SVM')
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.grid(color='b' , linewidth='0.1' ,linestyle = "-.")
plt.legend(loc='best')
plt.show()


#range4
range4=np.linspace(0.001,0.01,10)
score4=np.zeros(len(range4))

for j in range(len(range4)):
    clf1 = SVC(gamma=range4[j])
    clf1.fit(X_train, y_train)
    score4[j]=clf1.score(X_test,y_test)


plt.figure()
plt.plot(range4,score4)
plt.title('RBF kernel SVM')
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.grid(color='b' , linewidth='0.1' ,linestyle = "-.")
plt.legend(loc='best')
plt.show()

#print results
print('Best gamma = '+str(range4[np.argmax(score4)]))
print('Best Accuracy = '+ str(max(score4)))