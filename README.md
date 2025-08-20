📊 Loan Default Prediction  

![Python](https://img.shields.io/badge/Python-3.10-blue) 
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange) 
![Status](https://img.shields.io/badge/Status-Completed-brightgreen) 
![Dataset](https://img.shields.io/badge/Dataset-HMEQ-lightgrey)

---

 🔍 Project Overview  

This project focuses on predicting **loan defaults** using the **HMEQ dataset**.  
The dataset contains information about **loan applicants** including credit history, debt-to-income ratio, employment status, and more.  

The goal is to **identify high-risk borrowers** using machine learning models, helping financial institutions reduce **credit risk exposure**.  

---

 🎯 Objectives  

- Perform **Exploratory Data Analysis (EDA)** to uncover key trends.  
- Handle **missing values** and apply **feature engineering**.  
- Build and compare multiple machine learning models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
- Perform **hyperparameter tuning** for optimal performance.  
- Evaluate models using **Accuracy, ROC-AUC, Confusion Matrix, and Feature Importance**.  

---

 🛠️ Tech Stack  

- **Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Environment**: Jupyter Notebook  
- **Version Control**: GitHub  

---

 📊 Exploratory Data Analysis (EDA)  

Some key insights from the dataset:  

- Applicants with **poor credit history** show a much higher default rate.  
- **Debt-to-income ratio** is one of the strongest predictors of default.  
- Missing values in employment and mortgage variables were imputed with domain-specific strategies.  

---

 🤖 Model Performance  

| Model                 | Accuracy | ROC-AUC |
|------------------------|----------|---------|
| Logistic Regression    | 82%      | 0.75    |
| Decision Tree          | 85%      | 0.78    |
| Random Forest          | **88%**  | **0.81** |

👉 Random Forest performed the best overall.   

---

 🚀 How to Run  

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/loan-default-prediction.git
   cd loan-default-prediction
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run Jupyter Notebook  
   ```bash
   jupyter notebook
   ```

4. Explore notebooks inside `/notebooks/`.  

---

 ✅ Conclusion  

- **Random Forest** achieved the highest performance with **88% accuracy** and **0.81 AUC**.  
- The model can effectively **classify risky borrowers** and assist financial institutions in **decision-making**.  
- Future improvements could include **XGBoost/LightGBM**, **deep learning models**, and **deployment via Flask/Streamlit**.  

---

 📬 Contact  

👤 **Shatadru Bhattacharyya**  
💼 [LinkedIn](https://www.linkedin.com/in/shatadru-bhattacharyya-81428816/)  
📧 shatadru.b@gmail.com  
🌐 [Portfolio](https://github.com/shatadru-b)  
