

#  Email Spam Detection (Machine Learning)

This project is a simple **Email/SMS Spam Detection system** built using Machine Learning techniques and models. It was completed in **two days** and is my **first end-to-end ML project**, covering data preprocessing, feature extraction, model training, and evaluation.

The goal of this project is to classify messages as **Spam** or **Ham (Not Spam)** using text-based features and a supervised learning approach.

---

##  Project Overview

Spam messages are a common problem in email and messaging platforms. This project uses Natural Language Processing (NLP) techniques and a machine learning classifier to automatically identify spam messages.

Main features of the project:

* Text preprocessing and cleaning
* TF-IDF feature extraction
* Logistic Regression classification model
* Model evaluation using accuracy, confusion matrix, and classification report
* Visualization of results

---

##  Machine Learning Approach

### Dataset

The project uses a CSV dataset (`spam.csv`) containing labeled messages:

* **v1** → Label (`spam` or `ham`)
* **v2** → Message text

Steps applied to the dataset:

* Removed unnecessary columns
* Handled encoding issues (`latin-1`)
* Removed duplicate messages
* Converted labels into numeric form
* Checked class imbalance

---

### Text Processing

The following preprocessing steps were applied:

* Lowercasing text
* Removing stopwords
* TF-IDF vectorization

TF-IDF helps convert text into numerical features that represent word importance across messages.



### Model Used

* **Algorithm:** Logistic Regression
* **Reason:** Simple, fast, and effective for text classification tasks

The dataset was split into:

* **80% Training Data**
* **20% Testing Data**

as per usual standard (80/20)



## Model Evaluation

The model performance was evaluated using:

* **Accuracy Score**
* **Confusion Matrix**
* **Classification Report (Precision, Recall, F1-Score)**

Visualization includes:

* Class distribution plot
* Confusion matrix heatmap




##  Technologies & Libraries Used In The Code

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* NLTK



##  How To Run The Project (since I struggled with htese steps more than once, unfortunately)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/email-spam-detection.git
```

### 2. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

### 3. Open Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the Notebook

Open:

```
email-spam-detector.ipynb
```




## Learning Outcomes

From this project, I learned:

* How to preprocess text data
* How TF-IDF works for NLP tasks



## Room for Future Improvements

* Using advanced models like Naive Bayes, SVM, or Neural Networks
* Applying lemmatization and stemming
* Deploying the model using Streamlit or Flask
* Creating a web interface for real-time spam detection
* Hyperparameter tuning



## 👤 Author

**Farina Noman**
This is my First Machine Learning Project
Completed in 2 Days, yay :)


Just tell me 👍

