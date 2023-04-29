# Speech-Emotion-Recognition-Model
Develop a model which detect the emotion of speeches, which was trained by RAVDESS DataSet



ALGORITHMS USED--

1.	 Convolutional Neural Network (CNN)- 
      CNN requires the labeled training data to learn the patterns among input data, using this, it will make predictions. 
     In Speech Emotion Recognition, Audio Recordings which      are labeled with corresponding emotions can be given to the CNN as a labeled dataset for              training. So, by extracting features and classifying them into particular emotions, this can be self-learnt by the CNN.

2.	Random Forest Algorithm-
      This technique builds many decisions trees during the training phase and outputs the class that represents the mean of the classes(classification) or the     mean prediction(regression) of the individual trees.
    It handles missing values with both categorical and continuous data.
    
    
    
To run this model, you need to be installed with python and its path setup with cmd I had attached the requirements.txt files which contains all libraries need to be install to run the model successfully. You have to go to this "Speech-Emotion-Recognition-Model" folder on cmd if you are downloading zip file. then, type cmd " pip install requirements.txt " - This will install all libraries into the Python, If you are using Pycharm, then, you have to install the libraries manually from 'file->settings' menu in pycharm which I am specifying below.

Main Thing is the dataset, which I was provided in this repo

Python Libraries used in this Project –

1. NumPy-
      
          pip install numpy
 
2. Scikit-learn-

          pip install scikit-learn

3. Librose-

          pip install librose
      
4. Glob- 

          pip install glob
      
      
If there occurs the error of PREMISSION or any user access is needed while installing libraries - then use this command

	pip install library_name --user
  
  
This model gives you a output in array of different predicted emotions based on the training given to it for different audio formats by extracting multiple specifications of Audio files using RAVDESS dataset.

I had attached the .pdf which explains all about the Model with the steps which are need to be follow while making model code
