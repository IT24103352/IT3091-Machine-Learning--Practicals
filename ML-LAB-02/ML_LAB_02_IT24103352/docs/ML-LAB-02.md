# BSc (Hons) in Information Technology
## IT3091: Machine Learning
### Y3.S1 – 2026

**LAB SHEET 2**

# Data Understanding, EDA and Data Quality

## Learning Outcomes

By the end of this lab session, students will be able to

- Load and inspect a real-world dataset.
- Identify dataset characteristics.
- Identify feature types and the target variable.
- Perform basic Exploratory Data Analysis (EDA).
- Detect common data quality issues.
- Visualize data distributions and relationships.
- Perform train-test splitting correctly while avoiding data leakage.

## Dataset

- California Housing Dataset (given)

## Part A: Dataset Understanding

1. Load the California Housing Dataset and Display
   - First 5 records
   - Last 5 records
   - Shape
   - Column names
2. Identify number of samples, number of features and dataset type.
3. Identify the data set information by using **df.info()** and answer the following,
   - Which columns are numerical?
   - Are there categorical columns?
   - What is the target variable?
4. Display the statistical summary using and interpret the results.
   - Feature with the highest mean
   - Feature with the largest range
   - Feature with the largest standard deviation
   - Minimum value of each feature
   - Maximum value of each feature

## Part B: Exploratory Data Analysis

1. Check whether the dataset contains any missing values and duplicate records.
   (If duplicate records exist, remove them and display the updated dataset shape.)
2. Generate Histograms for the variables and identify,
   - Which feature looks normally distributed?
   - Which feature appears skewed?
   - Which feature contains the largest spread?
3. Choose an appropriate missing value imputation method for each variable based on its distribution and justify your choice.
4. Generate separate boxplots for each numerical feature and identify,
   - Which features contain outliers?
   - Why are outliers important in ML?
5. Create a scatter plot showing the relationship between Median Income vs Median House Value and interpret the plot.
6. Generate the correlation matrix (heatmap),
   - Which feature is most positively correlated with the target?
   - Which feature is least correlated?
   - Which two input variables have the strongest correlation?

## Part C – Train/Test Separation

1. By using **"train_test_split"** function split the dataset and display,
   - How many training samples?
   - How many testing samples?
2. Compare the distributions of the training and testing datasets using summary statistics and visualizations. Are the two datasets similarly distributed? Justify your answer.

## Part D – Understanding the concepts

1. Why is data understanding essential before training a machine learning model?
2. What is the difference between univariate and bivariate analysis?
3. Why should duplicate records be removed?
4. Why can missing values reduce model performance?
5. What is data leakage? Give one example.
6. Why must train-test splitting be performed before preprocessing?
