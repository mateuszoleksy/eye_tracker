# eye_tracker
Eye tracker project
Based on python
Main goal is to develop GOOD software driven eye tracker

My objectives are:
 - make eye recognition software

Based on:
 - labels, which are positions of the eyes in the single image in dataset

Status:
I made actually the CNN model in python to recognize eyes based on dataset, implemented the randomizer to the photos (it should also make changes to labels).
The problem with recognizing eyes is, if they stay in the same position (e.g. in the middle of the image). The model 'gets used to' this position.

The evaluation of model was done in WSL subsystem for linux