# 🧠 Exploring Mental Health

## 📋 Overview
This project was part of the Kaggle Playground Series (S4E11) competition. The goal was to predict the presence of depression in individuals using a dataset with mixed numerical and categorical features. The analysis involved data preprocessing, feature engineering, and the implementation of various machine learning models.

---

## 🎯 Key Results

- **Average ROC-AUC from 5-Fold Cross-Validation:** 0.9748
- **Training Metrics:**
  - **Accuracy:** 94.46%
  - **F1-Score:** 0.8454
  - **ROC-AUC:** 0.9802

---


## 🛠️ What Was Done

### Data Preprocessing
- **Outlier Removal:**
  - Removed outliers from the Age column (restricted range: 0-100).
- **Normalization:**
  - Converted categorical features to lowercase and removed whitespace.
- **Feature Splitting:**
  - Prepared separate datasets for features (X) and target variable (y).

### Feature Engineering
- **Numerical Features:**
  - Analyzed distributions and correlations.
- **Categorical Features:**
  - Encoded variables for model compatibility.

### Models Used
1. **Logistic Regression:** Balanced class weights to handle class imbalance.
2. **Random Forest Classifier:**
   - Applied class weights for balanced training.
3. **XGBoost Classifier:**
   - Tuned `scale_pos_weight` to address class imbalance.
4. **LightGBM Classifier:**
   - Automatically handled class weights with `balanced` parameter.

### Data Splitting
- Used **Stratified K-Fold Cross-Validation** (5 splits) to ensure robust model evaluation.

### Handling Class Imbalance
- Calculated class weights for the Depression variable to balance the dataset during training.

---

## 📊 Key Visualizations

1. **Histograms and Boxplots for Numerical Features:**
   - Visualized the distribution and potential outliers in numerical data.
     
![image](https://github.com/user-attachments/assets/429bb811-fe8a-4d76-9438-4c75b4d046cd)

![image](https://github.com/user-attachments/assets/9108fe3a-ae0a-41f0-ad60-480a6b121aaf)

![image](https://github.com/user-attachments/assets/6a0efd61-7282-47d5-8974-62e2d003b3c9)

  
2. **Bar Charts for Categorical Features:**
   - Highlighted the frequency of values in key categorical variables such as Profession, Degree, and Sleep Duration.

![image](https://github.com/user-attachments/assets/20d9e29d-b923-41b8-8a7e-d479a3fef4ad)

![image](https://github.com/user-attachments/assets/2c7cb3e8-def6-48b0-896b-d6f0068bc345)

    
3. **Distribution of the Target Variable (Depression):**
   - Showed class imbalance and calculated class weights for model training.
     
![image](https://github.com/user-attachments/assets/2c547db2-f8a6-4906-86e6-76d260d242fc)

  
5. **Correlation Matrix for Numerical Features:**
   - Displayed relationships between numerical variables.

![image](https://github.com/user-attachments/assets/5bb6c909-1451-4f52-8c16-6e896e0a72e9)


6. **Pairplots of Selected Features:**
   - Examined pairwise relationships for feature interaction.
  
     ![image](https://github.com/user-attachments/assets/859b080c-dd2c-4463-a904-f31307f9c50f)


---

## 🗂️ Repository Structure

```plaintext
├── train.csv                      # Training dataset
├── test.csv                       # Test dataset
├── sample_submission.csv          # Format for submission
├── Exploring Mental Data Health.ipynb # Jupyter notebook containing the analysis
├── README.md                      # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/exploring-mental-health.git
   cd exploring-mental-health
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook:

   ```bash
   jupyter notebook "Exploring Mental Data Health.ipynb"
   ```

4. Processed data and predictions will be saved in the specified output folder.

---

## 💡 Reflections and Future Work

- **Business Relevance:**
  - This analysis can support mental health professionals in identifying factors associated with depression.
  - The models provide a foundation for exploring data-driven mental health diagnostics.

- **Technical Learnings:**
  - Addressing class imbalance significantly improved model performance.
  - Visualization helped identify artifacts and relationships in the data.

- **Future Directions:**
  - Apply deep learning techniques to explore complex interactions.
  - Incorporate external datasets for enhanced generalization.

---

## 🤝 Contributions

If you have ideas or suggestions for improving this project, feel free to open an issue or submit a pull request. Contributions are welcome!

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🙏 Acknowledgments

- Kaggle for hosting the competition.
- The Kaggle community for providing a collaborative learning environment.
