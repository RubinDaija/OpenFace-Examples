import cv2
import facedetection as fd
import facerecognition as fr
import dlib
import comparer
import numpy as np
import sys
import glob


def getFaces(path):
	people = []
	files = glob.glob(path)
	for f in files:
	    name = (f.split('.'))[0]
	    kennypic = cv2.imread(f);
	    kenny = fd.detectFace(kennypic)
	    kenny = fr.getRepOfRect_rect(kenny,kennypic)
	    people.append([name,kenny])
	return people

people = [] #the individuals face vectors

res = None #this is to be used with the displaying of results


path = sys.argv[1]

people = getFaces(path)
img = cv2.imread(sys.argv[2])
rects = fd.detectFaces(img)
detects = fr.getRepOfRects_rect(rects,img)
res = comparer.compare(people,detects)





for name,i in res:
	cv2.rectangle(img,(i.left(),i.top()),(i.right(),i.bottom()),(0,255,0),2)
	cv2.putText(img,name,(i.left(),i.top() - 5), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,255,0),2)
cv2.namedWindow('imshow',cv2.WINDOW_NORMAL)
cv2.imshow("imshow",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
