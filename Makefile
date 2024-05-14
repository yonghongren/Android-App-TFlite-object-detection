
tflite-model-maker:
	conda create -n tensorflow pip python=3.9
	conda activate tensorflow
	# let tflite-model-maker pick and install required tensorflow (https://github.com/TnzTanim/Android-App-TFlite-object-detection)
	# pip install tensorflow
	pip install tflite-model-maker
	pip list # as of 12/22/2023 it shows tensorflow 2.9.3, which matches the log

train:
	python Train.py
