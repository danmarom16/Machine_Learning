# Machine Learning Project README

This repository contains the code for a machine learning project. The project focuses on training a neural network model using a dataset and making predictions on a test set. Follow the instructions below to set up the project and run the trained model.

## Set-up Instructions

To run the machine learning code in this project, follow these set-up instructions:

1. **Install Anaconda**: Download and install Anaconda, a popular Python distribution that includes various scientific computing libraries and tools. Anaconda can be downloaded from the official website: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual).

2. **Create a virtual environment**: Open a terminal or command prompt and create a new virtual environment using the following command:

3. **Activate the virtual environment**: Activate the virtual environment with the following command:
- For Windows:
  ```
  conda activate ml-project
  ```
- For Linux/Mac:
  ```
  source activate ml-project
  ```

4. **Install required libraries**: Install the necessary libraries by running the following command in the terminal:

5. **Download the code**: Download the code files from this repository and place them in a directory of your choice.

6. **Place your dataset in the same directory**: place your datasets in the same directory as the code files and make sure to call the train file - "train.csv", the validation file - "validation.csv" and the test file - test.csv".

7. **Run the code**: Open a Python IDE or a Jupyter Notebook and execute the code to run the machine learning project.

## Loading the Trained Model
If you want to use the existing code in the NN_final.ipynb to run the trained model, just run the last cell in the .ipynb file:<br><br>
![image](https://github.com/danmarom16/Machine_Learning/assets/92876036/a6ba1ba3-20be-4606-b941-9180afff331b)

<br>Else, if you want to load the trained model and make predictions with your costum test file, make sure you implement those essentials lines of code:

1. In your Python environment, create a new python script and include  import the necessary libraries:
```python
import pickle
import pandas as pd
import numpy as np
```

2. Load the trained model from the model.pkl file:
```python
with open('model.pkl', 'rb') as file:
    nn = pickle.load(file)
```

3. Load the test data from your test file (replace "YOUR_CSV_TEST_FILE.csv" with it's name) :
```python
test_data = pd.read_csv('YOUR_CSV_TEST_FILE.csv', header=None)
test_samples = test_data.iloc[:, 1:].values.T
```

4. Make predictions using the loaded model:
```python
test_predictions = nn.test(test_samples)
```
Important Note: Make sure your costum test file applies to the CSV file format described below.<br>

## Architecture And Formats
### Neural Network Architecture
The neural network architecture used in this project is as follows:

Input Layer: 3072 neurons (flattened vector of 32x32x3 image)<br>
* 1st Hidden Layer: 512 neurons
* 2nd Hidden Layer: 256 neurons
* Output Layer: 10 neurons (representing different classes)<br><br>
Note: The neural network implementation does not use TensorFlow or PyTorch. It is implemented from scratch using NumPy.

### CSV File Formats
The dataset files have the following formats:<br>
* aug_train.csv: Contains 8000 rows, each row representing a flattened vector of a 32x32x3 image. The first column of each row is the label of the photo.
* validation.csv: Contains 1000 rows, each row representing a flattened vector of a 32x32x3 image. The first column of each row is the label of the photo.
* test.csv: Contains 1000 rows, each row representing a flattened vector of a 32x32x3 image. The first column of each row is a question mark, which represents the placeholder where the model's test output will be placed per image.<br><br>
Note: The model should output its predictions and save them to the first column of the test.csv file.

## Code Breakdown
### Data Loading and Preprocessing
The dataset files (aug_train.csv, validate.csv, and test.csv) are loaded using the pandas library. The data is preprocessed to separate the samples (input features) and labels. The samples are transposed to have the shape (input_size, num_samples), and the labels are reshaped to (1, num_samples).

### Neural Network Class
The NeuralNetwork class implements a multi-layer neural network. It consists of methods for forward propagation, backward propagation, weight update, prediction, and accuracy calculation. The class is used for training the neural network model and making predictions.

### Training the Neural Network
The neural network model is created by instantiating an object of the NeuralNetwork class. The model is then trained using the gradient_descent method, which performs forward and backward propagation for a specified number of epochs. The training process includes updating the weights based on the calculated gradients.

### Testing the Neural Network
After training the model, the test method of the NeuralNetwork class is used to make predictions on the test samples. The predictions are obtained by performing forward propagation with the test data.

### Saving the Test Predictions
The test predictions are saved to the test.csv file by assigning them to the first column of the DataFrame obtained from pd.read_csv. The modified DataFrame is then saved back to the test.csv file using the to_csv method.

## Author
Dan Marom
