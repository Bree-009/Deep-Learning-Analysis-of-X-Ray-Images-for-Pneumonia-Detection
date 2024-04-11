# Deep-Learning-Analysis-of-X-Ray-Images-for-Pneumonia-Detection



![Alternative text](https://github.com/Bree-009/Deep-Learning-Analysis-of-X-Ray-Images-for-Pneumonia-Detection/blob/main/banner.png)



## BUSINESS UNDERSTANDING

The rapid advancement of Artificial Intelligence (AI) has brought significant benefits across various industries, including healthcare. Pneumonia, a critical lung infection, poses a serious threat, particularly to vulnerable demographics like the elderly and children. Traditionally, diagnosing pneumonia requires time-consuming physical examinations and lab tests, often necessitating multiple doctor visits.

To address this issue, we aim to develop a deep learning model capable of accurately detecting pneumonia from chest x-ray images. Such a tool holds immense value for healthcare professionals and patients, enabling quicker and more precise diagnoses. Radiologists and other specialists can leverage this technology to enhance their diagnostic accuracy, ultimately leading to better patient care and treatment outcomes.

Key stakeholders interested in leveraging deep learning for medical imaging include healthcare professionals, patients, hospitals, medical device manufacturers, and insurance companies. For instance, hospitals can optimize resource allocation and improve treatment efficacy, while medical device manufacturers can enhance product development for more accurate diagnoses. Additionally, researchers and government agencies stand to benefit from these advancements, using the models to deepen disease understanding and ensure regulatory compliance.

In summary, leveraging deep learning for medical imaging presents a transformative opportunity to enhance diagnostic accuracy, improve patient outcomes, and streamline healthcare processes across various stakeholders.





## 1.2. Technical Objectives

- Develop a deep learning model to accurately identify pneumonia from chest x-ray images.
- Fine-tune model architecture and parameters for optimal accuracy on validation data.
- Apply data augmentation techniques to enhance model generalization by expanding the training dataset.
- Experiment with various optimization methods and batch sizes to improve training efficiency and stability.
- Assess model performance using precision, recall, and F1 score metrics.




## 1.3. Business Objectives


- Provide pediatricians with a fast and precise tool for pneumonia diagnosis in children, potentially reducing unnecessary hospital visits and improving outcomes.
- Enhance access to pneumonia diagnosis in low-resource settings without immediate access to trained medical professionals.
- Potentially lower healthcare expenses by enabling early diagnosis and treatment of pneumonia in pediatric cases.
- Contribute to building a comprehensive dataset for pneumonia diagnosis, facilitating further research and model advancement.
- Develop a user-friendly model for seamless integration into existing hospital or clinic workflows, ensuring efficient and streamlined diagnosis processes.


## 1.4. Success Metrics

The performance of the model is evaluated based on:
* Accuracy - achieving an accuracy of over 85%
* Recall - achieving a recall of over 65%
* Loss Function - minimize loss function to ensure the model is not overfitting


## 1.4. Data Understanding


The dataset comprises 5,863 JPEG images categorized into two classes: "Pneumonia" and "Normal." It is organized into three main folders: "train," "test," and "val," each containing subfolders corresponding to the image categories.

The chest X-ray images, captured in the anterior-posterior view, were obtained from pediatric patients aged one to five years old at Guangzhou Women and Children’s Medical Center, Guangzhou. These images were part of routine clinical care procedures.

Before inclusion in the dataset, all chest radiographs underwent a quality control process to remove any low-quality or unreadable scans. Subsequently, the diagnoses assigned to the images were graded by two expert physicians. To mitigate potential grading errors, a third expert also evaluated the images in the evaluation set.

Overall, this dataset provides a curated collection of chest X-ray images, ensuring quality and accuracy through rigorous quality control measures and expert evaluation, making it suitable for training AI systems for pneumonia diagnosis.


## Data Visualization & Preprocessing

### EDA

This step involves examining and understanding the dataset before applying machine learning algorithms.

It will guide in processes like: data preprocessing, model selection, and performance evaluation strategies.

The EDA performed on the dataset includes visualizing samples of normal and pneumonia chest X-ray images, and plotting the class distributions for the training, test, and validation sets.


### Visualizing the distribution of classes in each category of data.

#### Visualizing the distribution of classes in the Train data

![Alternative text](https://github.com/Bree-009/Deep-Learning-Analysis-of-X-Ray-Images-for-Pneumonia-Detection/blob/main/images/training%20distribution.png)

Approximately 75% of the training data consists of normal chest X-rays while about 25% contains X-rays with pneumonia.


#### Visualizing the distribution of classes in the Test Data

![Alternative text](https://github.com/Bree-009/Deep-Learning-Analysis-of-X-Ray-Images-for-Pneumonia-Detection/blob/main/images/test%20distribution.png)

Approximately 63% of the Test Data consists of normal chest X-rays while almost 38% contains X-rays with pneumonia.



#### Visualizing the distribution of classes in the Validation Data
![Alternative text](https://github.com/Bree-009/Deep-Learning-Analysis-of-X-Ray-Images-for-Pneumonia-Detection/blob/main/images/validation%20distribution.png)

The distribution between the two classes (Pneumonia and Normal) is approximately even.


#### Analyzing the distribution of image sizes (width and height)

![Alternative text](https://github.com/Bree-009/Deep-Learning-Analysis-of-X-Ray-Images-for-Pneumonia-Detection/blob/main/images/scatterplot%20of%20image%20sizes.png)

The visualization above indicates that X-ray images of normal lungs are slightly larger in dimensions as compared to those of pneumonia-infected lungs.

## Data Preprocessing

### Extracting the images and labels
The images and labels were extracted from the train, test, and validation generators, and returned as arrays. The train_images array holds the image data, while the train_labels array contains corresponding labels.


#### Checking the information of the datasets
Information regarding the shape and size of both the image and label datasets was explored and displayed. Printing these values provides an overview of the dataset's sample count and the dimensions of the image and label arrays.
There are 5216 training samples, 624 testing samples, and 16 validation samples.
Each image in the datasets is represented as a 128x128x3 array (128x128 pixels with 3 color channels).
The labels for each dataset are represented as one-dimensional arrays with lengths corresponding to the number of samples in each dataset.


#### Reshaping the images.
The images were converted from their original 3-dimensional shape (height x width x channels) into a flattened format. This flattened representation allows the images to be easily processed and fed into machine learning models.
The image data arrays train_images, test_images, and val_images are reshaped into a 2D format where each row represents an image and each column represents a pixel value.

** Importance:**

1. Standardization of Input Format: Reshaping ensures that the input data conforms to the expected shape, making it compatible with the model's architecture.

2. Vectorization: Reshaping the images into a 2D format converts each image from a 2D or 3D array into a 1D array. Vectorization allows for easier processing and manipulation of the data.

3. Consistency: It ensures that all datasets have the same shape, making it easier to train and evaluate models consistently.

# Modeling

## Baseline model : A Densely Connected Neural Network

Initially, a baseline fully connected network is constructed utilizing the Keras Sequential API. This network consists of two hidden layers followed by an output layer. The first two layers incorporate ReLU activation functions, which introduce non-linearity to the network, enabling it to learn complex patterns in the data. Meanwhile, the final layer employs a sigmoid activation function, which outputs probabilities between 0 and 1, making it suitable for binary classification tasks.
This model consists of two hidden layers with ReLU activation and an output layer with sigmoid activation. The final training accuracy of 93.94% and validation accuracy of 87.50% suggest that the model has learned to classify the chest X-ray images reasonably well.
The baseline model (model_1) shows promising results, but the overfitting issue needs to be addressed through techniques such as regularization, dropout, or architectural modifications to improve the model's generalization capabilities and achieve better validation/test performance.


## Model 2: Convolutional Neural Network

The second model comprises a sequential architecture consisting of two convolutional layers, each followed by max pooling layers. These convolutional layers are responsible for extracting features from the input images, while the subsequent max pooling layers reduce the spatial dimensions of the feature maps. Finally, the model concludes with a dense layer, which performs the classification task.
- The convolutional neural network model (model_2) achieves a reasonable training accuracy of 80.41%, but its performance on the validation set is significantly worse, with a validation accuracy of only 56.25%.
- To improve the model's performance, further techniques such as regularization, dropout, or architectural modifications may be necessary to address the overfitting issue and enhance the model's ability to generalize to new, unseen data.

# Tuning CNN

## Model 3 - CNN with Architecture modifications
Adding More Layers
The architecture of the model was modified by adding more convolutional layers, increasing the number of filters in each layer and introducing two additional dense layers after the flattening layer.
This model further enhances the architecture by adding more convolutional and dense layers.
The model achieves a training accuracy of 98.66% and a validation accuracy of 75.00%.

# Model Evaluation

The performance of the two tuned models (Model 2 and Model 3) is compared using the following metrics:

- Test Loss
- Test Accuracy
- Test Precision

The performance metrics indicate that the best performing model (Model 3) has the following characteristics:

1. Validation Loss: 0.3279 - This low test loss suggests that the model is able to accurately predict the classes of the validation data, with minimal error.

2. Test Loss: 1.05 - The increased test loss suggests that the model is requires more training data to be able to accurately predict on unseen data.

3. Validation Accuracy: 0.75- The validation accuracy of 75% indicates that the model is able to correctly classify 75% of the validation samples, which is a good performance for a binary classification task.

4. Test Accuracy: 0.75 - The Model achieves a similar accuracy for test accuracy of 75% indicates that the model is able to correctly classify 75% of the test samples.

5. Test Precision: 0.708- The test precision of 0.7080 means that when the model predicts a sample as "Pneumonia", it is correct 70.80% of the time. This suggests the model has a relatively high precision in identifying positive (pneumonia) cases.

6. The results show that Model 3 outperforms Model 2 in all three metrics, with a lower test loss, higher test accuracy, and higher test precision.

![Alternative text](https://github.com/Bree-009/Deep-Learning-Analysis-of-X-Ray-Images-for-Pneumonia-Detection/blob/main/images/test%20loss%20and%20test%20accuracy.png)


![Alternative text](https://github.com/Bree-009/Deep-Learning-Analysis-of-X-Ray-Images-for-Pneumonia-Detection/blob/main/images/model%20accuracy%20plots.png)


![Alternative text](https://github.com/Bree-009/Deep-Learning-Analysis-of-X-Ray-Images-for-Pneumonia-Detection/blob/main/images/comparison%20of%20the%202%20models%20accuracy.png)



# Predictions

Overall, this processes tests image predictions by the chosed Model_3 based on a threshold and displays the images with their predicted class labels in a visually appealing grid format.
The performance metrics demonstrate that Model 4 is able to achieve a good balance between accuracy, precision, and low loss, indicating it is a robust and effective model for the pneumonia classification task.

# Conclusion

The developed deep learning models, particularly Model 3, have shown promising results in accurately detecting pneumonia from chest X-ray images. The models were able to achieve high training accuracy and reasonable validation/test performance, demonstrating their potential for real-world deployment in clinical settings.

The best performing model, "model_3", achieved a test Accuracy of 75% indicates that the model is able to correctly classify 75% of the test samples. This test accuracy although not bad could be improved by using more data such as larger and more diverse dataset of chest X-ray images.

# Recommendation

Based on the evaluation results, we recommend the deployment of Model 3 as the primary model for pneumonia diagnosis from chest X-rays. This model has exhibited the best overall performance, striking a balance between high accuracy, precision, and low loss. Its integration into hospital workflows and medical imaging analysis systems can significantly enhance the speed and accuracy of pneumonia diagnosis, ultimately leading to improved patient outcomes and more efficient healthcare processes.

Prospective Clinical Validation: Collaborate with healthcare providers to deploy the model in a real-world clinical setting and conduct a comprehensive evaluation of its performance, practicality, and impact on patient care.

# Next Steps

To further improve the model's performance and ensure its robustness, we suggest the following future work:

1. Expand the Dataset: Acquire a larger and more diverse dataset of chest X-ray images to improve the model's generalization capabilities and its ability to handle a wider range of pneumonia cases.
2. Pneumonia Severity Classification: Instead of just a binary classification (pneumonia vs. normal), extend the model to classify the severity of pneumonia (mild, moderate, severe). This would provide even more actionable insights for doctors.
3. Pneumonia Type Classification: Differentiate between bacterial and viral pneumonia. This information can guide treatment decisions as antibiotics are ineffective against viruses.
4. Localization of Pneumonia: Pinpoint the specific regions of the lung affected by pneumonia in the X-ray image. This can be crucial for further investigation and treatment planning.
5. Cloud-Based Deployment: Deploy the model on a cloud platform for remote access and scalability. This would allow for wider adoption and utilization from various hospitals and clinics.
