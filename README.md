# Face detection study
Comparison of different face detection algoritms (in python) tested on realworld footage, but a the face was printed on paper. 

Video files are available on my [drive](https://drive.google.com/file/d/1sDoxLUXXu6Ecsm8mXqaMpXq8SY3mOTKj/view?usp=sharing).

## Face detectors
Detectors used: Haar cascade, Hog detector (from *dlib* library), DNN (from *OpenCV* library).

## Conditions
The detectors were tested with 3 different lightning conditions: *Natural daily light*, *Dimmed daily light in the evening* and *Artificial industrial light*.

The tests were conducted under 5 different camera angles *(90°, 75°, 60°, 45°, 30°)*, where 90° means straight on.

## Measures
For the evaluation process the F1 score was used, which is defined as following: 
F1 = 2\*TP / (2\*TP+FP+FN)


------------------------------------------------------------------------------
Detailed info can be found in the [report](Report.pdf).
