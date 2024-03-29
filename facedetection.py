#!/usr/bin/python2.7
import dlib
import multiprocessing
import cv2

rects = multiprocessing.Queue()

def calcArea(rect1):
    return rect1.area()

# Calculate the overlaping percentage between twp rectengles
def overlap(rect1, rect2,A1,A2):
    rectOver = rect2.intersect(rect1)
    Aoverlap = rectOver.area()
    res = (float(Aoverlap) / (A1 + A2 - Aoverlap))
    return res

#This function detects the faces on the img, it uses up, left, right to know on
#how much to add to the rectangle values since the image on which the calculations
#are being done is not complete
#rects is used for outputing the results
def threaded_faceDetector(img,up,left,rects):
    detector = dlib.get_frontal_face_detector()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    dets2 = []
    for i,r in enumerate(dets):
        rect = dlib.rectangle(r.left() + left,r.top() + up,r.right() + left,r.bottom() + up)
        dets2.append(rect)
    rects.put(dets2)

#This function detects all the faces in the image using 4 threads
#The perMPfactor tell the function how much overlap the four sections of the picture
#should have, since a picture is divided into 4 parts one per thread
#the perMPfactor is multiplied with the amount of megapixels in the picture
def detectFaces(img,perMPfactor = 30):

    height,width,dimensions = img.shape
    extraAmount = height * width / 1000000 * perMPfactor

    imgq1 = img[0:(height/2) + extraAmount,0:(width/2)+extraAmount]
    imgq2 = img[(height/2) - extraAmount:height,0:(width/2)+extraAmount]
    imgq3 = img[0:(height/2) + extraAmount,(width/2)-extraAmount:width]
    imgq4 = img[(height/2) - extraAmount:height,(width/2)-extraAmount:width]

    jobs = []
    j = multiprocessing.Process(target = threaded_faceDetector, args = (imgq1,0,0,rects))
    jobs.append(j)
    j = multiprocessing.Process(target = threaded_faceDetector, args = (imgq2,(height / 2) - extraAmount - 1,0,rects))
    jobs.append(j)
    j = multiprocessing.Process(target = threaded_faceDetector, args = (imgq3,0,(width / 2) - extraAmount - 1,rects))
    jobs.append(j)
    j = multiprocessing.Process(target = threaded_faceDetector, args = (imgq4,(height / 2) - extraAmount - 1,(width / 2) - extraAmount - 1,rects))
    jobs.append(j)


    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    res = [rects.get() for p in jobs]

    rectangles = list()
    for i in res:
        for j in i:
            rectangles.append(j)


    i = 0
    while i < len(rectangles)-1:
        A1 = calcArea(rectangles[i])
        j = i + 1
        while j < len(rectangles):
            A2 = calcArea(rectangles[j])
            overRat = overlap(rectangles[i],rectangles[j],A1,A2)

            if overRat > 0.3: #if two rectangles have an overlaping area bigger than 30% they are discarded
                if A1 >= A2:
                    rectangles.pop(j)
                    j = j - 1

                else:
                    rectangles.pop(i)
                    i = i - 1
                    break
            j = j + 1
        i = i + 1


    return rectangles

# This function recognizes the biggest face in the picture
def detectFace(img):
    detector = dlib.get_frontal_face_detector()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    result = None
    maxArea = 0
    for i,r in enumerate(dets):
        rect = dlib.rectangle(r.left(),r.top(),r.right(),r.bottom())
        if rect.area() > maxArea:
            result = rect
    return result
