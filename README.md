# OpenFace-Example

This example is based from the examples provided in the OpenFace demos, specifically https://github.com/cmusatyalab/openface/blob/master/demos/compare.py . Hence it is more or less the same example, however I tried to comment most of the code so that it is easier to understand. Furthermore this example can detect multiple faces in a picture and identify individuals using the L2 distance calculations. In addition to that the face detection is parallelized for 4 threads, as it was first implemented in a Raspberry Pi. However in order for the code to be capable of identifying the individuals you have to provide the faces as individual pictures and then the group picture. The code can be run as:
  python main.py 'individualFacesPath' groupImage
-individualFacesPath: must be between apostrophes and can include a bunch of pictures simply by writing 'path/*.jpg'
-groupImage: is without apostrophe and one image is expected

### Comparison
The comparison to identify individuals is done on the basis of who has the smallest L2 distance based on the vector calculations. Thus the code calculates the face vectors of everyone found in the picture with everyone in the database. Then it picks the best result for each individual. It starts marking individuals as identified from the one with the smalles L2 calculated distance. However if two individuals are detected at the same person in the picture then the one with the smallest value is selected and the other one chooses another contender from the people in the picture who are not yet marked. Furthermore the compare code has a low and a high threshold. That is such that if an individual detected has an image size which is less than 96x96 then we use the high threshold value, otherwise we go with the low one. The 96x96 is used since that is the dimension at which the pictures are set up for recognition.

On how to set up OpenFace you can follow the steps on the pdf. The steps work for both a ARM based system and an X86 based system. However note that the ARM system has drasctically lower performance than the x86 based one.


For more information please visit:
- https://cmusatyalab.github.io/openface/ 
- https://github.com/cmusatyalab/openface 
