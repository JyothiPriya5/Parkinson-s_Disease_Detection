# 🧠 Parkinson's Disease Detection
## Introduction
**Parkinson’s Disease** is a progressive neurological disorder that primarily affects movement. It occurs due to the gradual loss of **dopamine-producing neurons** in the brain, leading to symptoms like tremors, stiffness, slow movement (bradykinesia), and balance issues. Though the exact cause is unknown, genetics and environmental factors may play a role. While there is no cure, treatments like medications, therapy, and deep brain stimulation help manage symptoms.

## 📌 Project Overview  
This project provides a **comparative analysis** of **Parkinson's Disease Detection** using **seven different Machine Learning Algorithms** for early diagnosis. It analyzes **biomedical voice measurements** and predicts whether a person has Parkinson’s disease based on given features. The **Random Forest model achieved the highest F1-score of 96.15%** after **hyperparameter tuning**. This project also integrates **MongoDB for data storage** and includes a **Flask-based web application** for real-time predictions.

## 📂 Dataset  
- Sourced from the **UCI Machine Learning Repository**.  
- Contains **voice recordings** from both Parkinson’s patients and healthy individuals.  
- Features include **MDVP (pitch-related measures), jitter, shimmer, HNR, status, and spread parameters**.  

## 🛠️ Technologies Used  
- **Python**  
- **MongoDB** (for storing patient data and predictions)  
- **Scikit-learn** (for machine learning)  
- **Flask** (for building the web interface)  
- **Matplotlib & Seaborn** (for data visualization)  
- **Jupyter Notebook / Google Colab**  

## 🏗️ Model Training  
1️⃣ **Data Preprocessing**  
   - Feature normalization using **MinMaxScaler**.  
   - Handled missing values and performed feature selection.  

2️⃣ **Machine Learning Models Compared**  
   - **Decision Tree (DT)**  
   - **Logistic Regression (LR)**  
   - **Support Vector Machine (SVM)**  
   - **Random Forest (RF)**  
   - **XGBoost**  
   - **K-Nearest Neighbors (KNN)**  
   - **Naive Bayes**  

3️⃣ **Hyperparameter Tuning**  
   - Optimized model parameters using **GridSearchCV**.  

4️⃣ **Database Integration**  
   - Stored patient details and predictions in **MongoDB**.  

5️⃣ **Web Application**  
   - Developed a **Flask-based UI** for real-time predictions.  
   - Evaluated models based on **accuracy, precision, recall, and F1-score**.  

## 🔍 Model Performance  
Even though **Random Forest, SVM, and KNN** had same accuracy, the **best-performing model** was **Random Forest**, achieving:  
- **Accuracy**: **96.61%**  
- **F1-Score**: **96.15%**

   ![image](https://github.com/user-attachments/assets/6b19da12-83ca-471f-922e-96b029010eb6)

**Feature Importance Analysis:**  
- **Jitter, Shimmer, and HNR** significantly contribute to predictions.   

## 🚀 How to Run  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/JyothiPriya5/Parkinson-s_Disease_Detection.git
cd Parkinson-s_Disease_Detection
```
### 2️⃣ Set Up MongoDB
- Install MongoDB and start the server:
```bash
mongod --dbpath <your-db-path>
```
- Configure the database connection in config.py.
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Run the Model in Jupyter Notebook
```bash
jupyter notebook Parkinsons_Disease_Detection.ipynb
```
5️⃣ Run the Flask Web App
```bash
python app.py
```
- Access the Flask web app at: http://localhost:5000.

### 📌 Future Improvements
- 🔹 Implement deep learning models for better accuracy.
- 🔹 Use larger datasets for improved generalization.
- 🔹 Deploy the model using FastAPI for production-ready performance.

