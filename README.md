# cs231n-Assignments
My implementations of stanford cs231n: Convolutional Neural Networks for Visual Recognition(Spring 2017) assignments<br><br>
I've changed some of the provided codes which can't work properly on my computer:<br>
* In assignment 3, the codes provided to show the coco dataset can't work and returned a [WinError 32]: "The process cannot access the file because it is being used by another process.". So I changed the function 'image_from_url' in the file image_utils.py.
* In assignment 3, in the file 'NetworkVisualization-TensorFlow.ipynb' and the file 'StyleTransfer.ipynb', the codes provided to load squeezenet can't work on my computer. I commented out two lines of codes in cell 2 that are used to detect the file path.
* In assignment 3, in the file 'LSTM_Captioning.ipynb'. In the 'Extra Credit' part, function BLEU_score and evaluate_model can not run properly on my computer. I've done a few changes on those codes.
