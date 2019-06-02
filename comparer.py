import numpy as np
import pyximport
pyximport.install(language_level=2)

HIGHTHRESH = 1.5
STANDARDTHRESH = 1.0

#faces have ID,faceVector ; observed faces a list of faceVectors,rects
def compare(listStudentFaces, listOfObservedFaces):
    comparedFaces = {} #a dictionary that will hold a comparisons for a face except of the best result
    bestResultPerStudent = [] # alist of the best results per person [valueOfComparison, theStudentID ,idOfObservedFace, rectangleOfObservedFace]
    #go through the individuals faces
    for i in listStudentFaces:
        fvec = i[1]#get the faceVector for current person
        tmpList = [] #the calculation per person that will be added to compared faces
        j = 0 #j will be used as an ID of observed faces hence they will all will have ID range [0-N)
        #go through all the observed faces
        while j < len(listOfObservedFaces):
            #get the comparison value
            d = fvec - listOfObservedFaces[j][0]
            result = np.dot(d,d)
            tmpList.append([result,j,listOfObservedFaces[j][1]])
            j = j + 1
        #sort the observed faces compared with the current person based on their valueOfComparison; sorted based on increasing order
        tmpList = sorted(tmpList, key=lambda x : x[0])
        #remove the best result
        tmp = tmpList.pop(0)
        #put the other results of that current person in the dictionary if they are later needed
        comparedFaces[i[0]] = tmpList
        #put that best result in the list that will be used for recognition
        bestResultPerStudent.append([tmp[0],i[0],tmp[1],tmp[2]])
    #sort the best result for each person increasing order
    bestResultPerStudent = sorted(bestResultPerStudent, key=lambda x : x[0])
    usedObservs = set() #will hold the IDs of  already recognized faces on the picture
    detections = list() #will hold the studentId recognized and the rectangle where it was recognized
    listLength = len(listOfObservedFaces)
    counter = 0
    while counter < listLength and 0 != len(bestResultPerStudent): #can not detect more faces than the recognized ones
        #get the best current observation, remove it also from the list
        i = bestResultPerStudent.pop(0)
        #if that observed face has not yet been recognized
        if i[2] not in usedObservs:
            #if the size of the detection is smaller than the threshold then use the high threshold
            if i[3].height() <= 96 or i[3].width() <= 96:
                if i[0] <= HIGHTHRESH:
                    usedObservs.add(i[2]) #add recognized face to the used set
                    detections.append([i[1],i[3]]) #add the detection to the results
                    counter = counter + 1 #if we recognize a person then increment the count
            else:
                if i[0] <= STANDARDTHRESH:
                    usedObservs.add(i[2]) #add recognized to the used set
                    detections.append([i[1],i[3]]) #add the detection to the results
                    counter = counter + 1 #if we recognize the person increment count
        #if the recognized face has been used
        else:
            #get the other results for the person that had a conflict
            tmpL = comparedFaces[i[1]]
            #get the next best value
            newVal = tmpL.pop(0)
            #if the current smallest is bigger than the HIGHTHRESH then ignore this person
            if newVal[0] <= HIGHTHRESH:
                #add it back to the list
                bestResultPerStudent.append([newVal[0],i[1],newVal[1],newVal[2]])
                #resort the list again so that the best place is found for the new value
                bestResultPerStudent = sorted(bestResultPerStudent, key=lambda x : x[0])

    #return a list of ID and rects of recognized faces
    return detections
