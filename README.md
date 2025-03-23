# 🧠 Exploring Mental Health

## 📋 Overview
This project was part of the Kaggle Playground Series (S4E11) competition. The objective was to predict depression among individuals using a dataset that includes both numerical and categorical features. The dataset posed several challenges:

- **Incomplete Data**: Several features had substantial missing values (e.g., Academic Pressure, Study Satisfaction, CGPA).
- **Class Imbalance**: The dataset was heavily skewed towards non-depressed individuals (81.8% with no depression), requiring techniques like class weighting.
- **Mixed Data Formats**: The dataset contained both classification-based (.csv) and potentially time-series data (e.g., Work/Study Hours, Financial Stress).
- **Data Cleaning Issues**: The dataset required standardization of categorical values, handling of outliers, and dealing with inconsistent labeling.

The analysis involved:

- **Data Exploration**: Identification of missing values, distribution analysis, and correlation heatmaps.
- **Data Preparation**: Normalization of features and handling of categorical variables.
- **Model Training**: Comparison of different models with strategies for handling missing values and class imbalance.
- **Visualization**: In-depth data visualizations to uncover patterns.
- **Technologies Used**: Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn.

---

## 🎯 Key Results

- **Average ROC-AUC from 5-Fold Cross-Validation:** 0.9748
- **Training Metrics:**
  - **Accuracy:** 94.46%
  - **ROC-AUC:** 0.9802
- **Class Imbalance Handling:** Improved the recall for the minority class without sacrificing overall performance.

---

## 🛠️ What Was Done

### Data Preprocessing
- **Outlier Removal:**
  - Removed outliers in the `Age` column based on a valid range of [18, 60].
  
  ```python
  train_data = train_data[(train_data['Age'] >= 18) & (train_data['Age'] <= 60)]
  ```

- **Normalization of Categorical Data:**
  - Standardized categorical values by converting them to lowercase and stripping whitespace.

  ```python
  train_data['Profession'] = train_data['Profession'].str.lower().str.strip()
  ```

- **Splitting Features and Labels:**
  - Prepared `X_train`, `y_train`, and `X_test` datasets.

  ```python
  X_train = train_data.drop(columns=['Depression'])
  y_train = train_data['Depression']
  ```

### Addressing Class Imbalance
- **Computed Class Weights:**
  
  ```python
  class_weights = compute_class_weight(
      class_weight='balanced',
      classes=np.array([0, 1]),
      y=y_train
  )
  ```

  - Applied these weights in models such as Logistic Regression and Random Forest.

### Models Used

1. **Logistic Regression:**
   - Adjusted class weights to balance the dataset.

   ```python
   logistic_model = LogisticRegression(class_weight=class_weight_dict, random_state=42)
   ```

2. **Random Forest Classifier:**
   - Incorporated class weighting during training.

   ```python
   rf_model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
   ```

3. **XGBoost Classifier:**
   - Used `scale_pos_weight` to mitigate imbalance.

   ```python
   scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]
   xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
   ```

4. **LightGBM Classifier:**
   - Leveraged the built-in `balanced` class weight parameter.

   ```python
   lgbm_model = LGBMClassifier(class_weight='balanced', random_state=42)
   ```

### Cross-Validation
- Used **Stratified K-Fold Cross-Validation** (5 splits) for reliable performance assessment:

  ```python
  cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
  print(f"Average ROC-AUC: {cv_scores.mean():.4f}")
  ```

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


---

## 🗂️ Repository Structure

```plaintext
├── train.csv                      # Training dataset
├── test.csv                       # Test dataset
├── sample_submission.csv          # Format for submission
├── Exploring Mental Data Health AI.ipynb # Jupyter notebook containing the analysis
├── README.md                      # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/SSJ0406/Exploring-Mental-Health-AI.git
   cd Exploring-Mental-Health-AI
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
