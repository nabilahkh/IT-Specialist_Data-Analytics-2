# IT SPECIALST : DATA ANALYST 2

### Type of Data Analysis
#### • Descriptive
Descriptive analytics involves the statistical analysis of past data to uncover patterns and correlations. It aims to describe events, phenomena, or outcomes, providing insights into past occurrences and serving as a foundation for trend analysis in business.

In descriptive analytics, several things can be used in solving:
- Metrics (Count, Mean, Mode, Std deviation, Min, Max, Avg, Sum, unique values)
- Searching
- Filtering
- Interpreting Result

#### • Metrics
Function describe()

                                                    df.describe()

- Count: The total number of elements in the column
- Mean: The mean average of the element in the column.
- Std: The standard deviation is the square root of the average of the squared deviations from the mean
- Min: The smallest value of the element in the column
- Max: The largest_value of the element in the column
- Quartiles: Values that divide a dataset into four equal parts, providing insights into the spread and distribution of the data.

#### • Diagnostic
Diagnostic analytics is a form of data analysis used to understand the reasons behind occurrences. It delves into trends and relationships among variables to pinpoint the underlying causes. This type of analysis follows descriptive analytics, which focuses on identifying what happened.

Several things can help with diagnostic analytics:
- Data drilling
- Data mining
- Data Relationship

  
### Data Aggregation and Interpretation Metrics
Data aggregation involves combining and summarizing individual data points into a larger dataset, while interpretation metrics are measures used to make sense of aggregated data, providing insights and understanding.

                                                    df.describe()
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/453d9981-3c06-4881-80b1-4c367cd1ed29" /></div>

### Interpretation: Storytelling
In 1930, the prestigious football competition known as the "World Cup" was held for the first time, marking an event eagerly anticipated worldwide until 2014, the year covered by this dataset. Throughout this period, the World Cup has been a highly anticipated event, with numerous matches played in stadiums across the globe.

Over the years, from 1930 to 2014, the average number of goals scored by the home team was 1.82, with a standard deviation of 1.62, while the visiting team scored an average of 1.02 goals, with a standard deviation of 1.07. The highest number of goals scored by the home team in a single match was 10, compared to 7 goals by the visiting team.

On average, nearly 45,000 spectators attended each match, filling stadiums worldwide to enjoy the games and support their favorite teams. Analysis of halftime performance revealed that the home team had an advantage in scoring with an average of 0.71 goals per half, with a standard deviation of approximately 0.94 goals. The record for the most goals scored by the home team in the first half of a match was 6.

Conversely, the visiting team scored an average of 0.42 goals per half, with a standard deviation of approximately 0.67 goals. The highest-recorded number of goals by the visiting team in the first half was 5.

Overall, the football competition during this period experienced rapid growth, with matches starting in 1930 and reaching their peak in 2014. However, there was variation in the number of matches each year, with the earlier period (1930-1960) featuring around 10 matches per year, while in recent years, the number has surged to over 50 matches per year. This reflects the evolution and changes in the dynamics of football competition over the past few decades.

### Exploratory Data Analysis Methods

Import Python Libraries

                                                import pandas as pd
                                                import numpy as np
                                                import matplotlib.pyplot as plt
                                                import seaborn as sns
                                                
                                                import warnings 
                                                warnings.filterwarnings('ignore')
                
                                                # Load dataset
                                                df = pd.read_csv('Titanic.csv')
                
                                                # Preview dataset
                                                df.head()
                
                                                df.info()

                                
#### Data Cleansing

                                                # Check missing value
                                                df.isna().sum()
                
                                                # Fill missing value for column 'Age'
                                                val = df.Age.median()
                                                df['Age'] = df.Age.fillna(val)
                
                                                # Fill missing value for column 'Embarked'
                                                val = df.Embarked.median()
                                                df['Embarked'] = df.Embarked.fillna(val)
                
                                                # Drop column 'Cabin'
                                                df.drop('Cabin', axis=1, inplace=True)
                                
### Exploratory Data Analysis (EDA)
involves several methods to delve into datasets and uncover valuable insights:

#### 1. Identifying Data Relationships
EDA helps identify relationships between different variables in the dataset. This can be done through visualizations such as scatter plots, correlation matrices, and heatmaps, which reveal how variables are related to each other.

#### 2. Describing Data Drilling Concepts
Data drilling involves examining data at different levels of granularity. Granularity refers to the level of detail or specificity in the data. EDA explores data at various granularities to understand patterns and trends. For example, drilling down from yearly data to monthly or daily data provides a more detailed perspective.

#### 3. Describing Data Mining Concepts:
##### Correlation Analysis
EDA examines correlations between variables to identify patterns and relationships. Correlation matrices and scatter plots are commonly used to visualize correlations.

                                                
                                                df_survived_notnull = df['Survived'][df['Survived'].notnull()]
                                                df_age_notnull = df['Age'][df['Age'].notnull()]
                                                df_pclass_notnull = df['Pclass'][df['Pclass'].notnull()]

                                                df1 = df[['Survived', 'Age', 'Pclass']]
                                                plt.figure(figsize = (10,8))
                                                sns.heatmap(df1.corr(), annot=True);

<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/b92917f8-36a4-4173-8ce6-a24623db3e80" /></div>


#### • Anomalies
- Anomalies are data or values that deviate from the expected pattern in a dataset.
- They may include nonsensical values, such as negative ages or fractional quantities, which are unrealistic in real-world scenarios, such as negative age (-15 years), which are implausible in real-world contexts.
- Anomalies can be detected through exploratory data analysis (EDA) techniques, such as visualizations like box plots and histograms.
  
#### • Outliers
- Outliers are data points that significantly differ from the majority of the dataset. Such as an age of 80 years is not inherently unreasonable in the context of human age, but its value is significantly distant from the general average, making it an outlier.
- They represent notable deviations from the dataset's norm and may indicate interesting insights or data errors.
- Outliers are identified using EDA methods like box plots, scatter plots, and z-score analysis.
  
**Anomaly is an outlier HOWEVER not all outliers are anomalies.**

##### Outliers
Outliers don't always need to be discarded; they can be reasonable but fall into the outlier category due to their limited quantity.

##### Anomalies
Anomalies must be removed as they deviate significantly from the expected pattern or are logically implausible.

                                                df_age_notnull = df['Age'][df['Age'].notnull()]
                                                
Finding first quantile and third quantile

                                                q1, q3 = np.percentile(df_age_notnull,[25,75])
                                                print(q1, q3)
                                                
Find the IQR which is the difference between third and first quartile

                                                iqr = q3 - q1
                                                print(iqr)
                                                
Find lower and upper bound

                                                lower_bound = q1 - (1.5 * iqr)
                                                upper_bound = q3 - (1.5 * iqr)
                                                
                                                print(lower_bound)
                                                print(upper_bound)
                                                
##### Handling Outlier

                                  df_age_new = df['Age'][(df['Age'] > lower_bound) & (df['Age'] < upper_bound)]
                                  print(df_age_new)
                                  df_age_new.describe()
                                  sns.boxplot(x=df_age_new);

<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/39dd468c-e04d-49a4-943e-2676215aae43" /></div>


### Hypothesis Testing
A statistical hypothesis test is a procedure in statistical analysis aimed at determining whether the available data provides enough evidence to support a specific hypothesis. This process usually entails computing a test statistic based on the data. Subsequently, a decision is made by comparing the test statistic to a critical value or by assessing a p-value derived from the test statistic.

#### • T-test
The t-test assesses the difference in means between two groups of data. It's a hypothesis test conducted using random samples from each group. Through this test, analysts determine if the same treatment yields consistent results in both groups or if there are differences.
Accepted hypotheses are:
Ho: No difference between the groups.
Ha: Difference exists despite the same treatment

#### • Z-test
The z-test compares means or proportions between two groups when the sample size is large (typically n > 30) and the population standard deviation is known. It's akin to the t-test but relies on the standard normal distribution (Z-distribution). Commonly used for hypothesis testing when the population standard deviation is known.

Accepted hypotheses are:
- Ho: There is no significant difference between the groups.
- Ha: A significant difference exists despite the same treatment.

##### Import Python Libraries
                                    import pandas as pd
                                    from scipy import stats
                                    from statsmodels.stats import weightstats as stests
                                    
An experiment on the `effects of anti-anxiety medicine on memory recall when being primed with happy or sad memories`. The participants were done on novel Islanders whom mimic real-life humans in response to external factors.

Drugs of interest (known-as) [Dosage 1, 2, 3]:
A - Alprazolam (Xanax, Long-term) [1mg/3mg/5mg]
T - Triazolam (Halcion, Short-term) [0.25mg/0.5mg/0.75mg]
S- Sugar Tablet (Placebo) [1 tab/2tabs/3tabs]

                                                    df.head()
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/69399bdb-23bb-41f8-b86c-7ea2e1292c5a" /></div>

- first_name : First name of Islander
- last_name : Last. name of Islander
- age : Age of Islander
- Happy_Saf_group : Happy or. Sad Memory priming block
- Dosage : 1-3 to indicate the level of dosage (low - medium - over recommended daily intake)
- Drug : Type of Drug administered to Islander
- Mem_Score_Before : Seconds - how long it took to finish a memory test before drug exposure
- Mem_Score_After : Seconds - how long it took to finish a memory test after addiction achieved
- Diff : Seconds - difference between memory score before and after

                                      df[['Mem_Score_Before', 'Mem_Score_After']].describe()
                                      df.head()
                                      df.shape
  

#### • T-test

                                      ttest,pval = stats.ttest_rel(df['Mem_Score_Before'], df['Mem_Score_After'])
                                      print(pval)
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/85e35475-67d1-4138-8433-f825f3da4391" /></div>    

                                      if pval<0.05:
                                          print("Reject null hypothesis")
                                      else:
                                          print("Accept null hypothesis")
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/daaf5317-bb12-4dff-8bac-879690a5ad73" /></div>                                              
At a significance level of 5%, drug exposure has had a significant impact on the time required to complete the memory test.


### Machine Learning Regression - Simple Linear Regression
Simple Linear Regression is a statistical method used to model the relationship between a single independent variabel and a dependent variable by fitting a linar equation to the observed data.

#### • Study Case

We use “Salary_Data” dataset to involves predicting about the salary or wages of workers or individuals based on experience
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/b1d35f95-8396-47e8-a879-154aa66aa953" /></div>   

##### Load and Check Data
###### Import Library

                                      import pandas as pd
                                      import numpy as np
                                      import matplotlib.pyplot as plt
                                      import seaborn as sns
                                      from sklearn.linear_model import LinearRegression
                                      from sklearn.model_selection import train_test_split
                                      from sklearn.preprocessing import StandardScaler
                                      from sklearn.preprocessing import MinMaxScaler
                                      from sklearn.metrics import mean_absolute_error
                                      from sklearn.metrics import mean_absolute_percentage_error
                                      from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
                                      
                                      import warnings
                                      warnings.filterwarnings('ignore')

###### Load Dataset
                                                # Load dataset
                                                df = pd.read_csv('Titanic.csv')
                
                                                # Preview dataset
                                                df.head()
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/36691cdd-e583-4a15-92b0-0efd69d44d08" /></div>

###### Check Data Condition
                                                    df.head()
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/af858379-7056-49f4-8c0b-acea458d6406" /></div>                                                

                                                    df.isna().sum()
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/4296ac93-6789-40fb-8750-cef10c05f378" /></div>

**Data is Clean**

### Explanatory Data Analysis

###### Distribution of YearsExperience
                                                    sns.distplot(df['YearsExperience']);
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/52cf54e3-de4c-491d-80dc-17bb9ea13629" /></div>

###### Distribution of Salary
                                                    sns.distplot(df['Salary']);
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/da0f6a38-3612-4677-9262-79157944c64c" /></div>


### Simple Linear Regression

###### Processing Modeling

                                                    x = df.drop(['Salary'], axis=1)
                                                    y = df['Salary']

###### Splitting, Training & Test Set

                              x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 1/3, random_state = 42)

###### Fitting Into Training
                                                    regressor = LinearRegression()
                                                    regressor.fit(x_train, y_train)
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/dc1b4bdf-6fd2-4ba2-aa23-a4f87cb84230" /></div>                                                  

                                                    regressor.coef_
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/94eb8b56-8bec-4939-91f7-9a967f372081" /></div> 

###### Predict the result 
                                                    y_pred = regressor.predict(x_test)

###### Plot The Result “Bar”
                                                    result = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
                                                    result
                                                    result.plot(kind = 'bar', figsize = (10,8))
plt.show()
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/359de4ec-af4a-4e02-ba56-f48c26643da9" /></div>

                                                    plt.scatter(x_train, y_train, color='red')
                                                    plt.plot(x_test.values, y_pred, color='blue')
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/b4470f9c-a09c-427d-96e4-42c364c1f18d" /></div>                                                    

#### Evaluate Model
#####  • Mean Squared Error (MSE)
                                                    np.sqrt(mean_squared_error(y_test, y_pred))
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/bdd31050-f056-4a31-a3f8-98ea9920c5b4" /></div>   
                                                    
is a regression model evaluation metric that is used to calculate the error between the values predicted by the model and the actual values. MSE is the average of the squared differences between the predicted value and the actual value.

#####  • Mean Absoulute Error (MAE)
                                                    mean_absolute_error(y_test, y_pred)
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/d6aba859-fe0d-4f19-a17b-051a7add2ffa" /></div>   

                                                    mean_absolute_percentage_error(y_test, y_pred)
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/7bb6dea1-db6e-4f96-a42f-050e1fa429e3" /></div>  

is a regression model evaluation metric that is used to calculate the error between the values predicted by the model and the actual values.

#####  • R-Squared (R2)
                                                    r2_score(y_test, y_pred)
<div align="center"><img src="https://github.com/nabilahkh/IT-Specialist_Data-Analytics-2/assets/92252191/366c1341-dd13-439b-94ff-f0b1f7ba2d32" /></div>

R-squared or R² is one of the regression model evaluation metrics used to calculate the level of success of the model in describing the variance of the target variable.
