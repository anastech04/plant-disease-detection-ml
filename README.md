\# Plant Disease Detection Using Machine Learning



This project is a computer vision and machine learning project that classifies plant leaves as healthy or diseased. The model can also identify the specific disease based on leaf images.



\## Project Overview



Early plant disease detection is important to prevent crop loss and reduce the overuse of pesticides. This project uses deep learning to automatically classify plant leaf images and support faster disease detection.



\## Features



\- Plant leaf image classification

\- Healthy vs diseased leaf detection

\- Disease category prediction

\- Image preprocessing and cleaning

\- Data augmentation to reduce overfitting

\- Model evaluation using accuracy, precision, recall, and F1-score



\## Technologies Used



\- Python

\- TensorFlow / Keras

\- EfficientNetB0

\- NumPy

\- Matplotlib

\- Scikit-learn

\- OpenCV



\## Dataset



The project uses a plant leaf image dataset.  

The dataset is not uploaded to this repository because of its large size.



\## Preprocessing



The preprocessing steps include:



\- Removing missing or corrupted images

\- Removing duplicate images

\- Splitting the dataset

\- Normalizing pixel values

\- Applying data augmentation such as rotation, flipping, zooming, and brightness adjustment



\## Model



The final model uses EfficientNetB0 with transfer learning. EfficientNetB0 was selected because it performed better than ResNet50 and handled image features more effectively.



\## Results



The final model achieved:



\- Test Accuracy: 96.15%

\- Weighted Precision: 94.1%

\- Weighted Recall: 93.8%

\- Weighted F1-score: 93.9%



\## How to Run



Install the required libraries:



pip install -r requirements.txt



Prepare and split the dataset:



python split\_data.py



Apply data augmentation:



python augment\_data.py



Train the model:



python train\_efficientnet.py



Test the model:



python test\_efficientnet.py



\## Author



Anas Abdallah

