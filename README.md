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

                                
Data Cleansing

Exploratory Data Analysis (EDA)
involves several methods to delve into datasets and uncover valuable insights:
1. Identifying Data Relationships
EDA helps identify relationships between different variables in the dataset. This can be done through visualizations such as scatter plots, correlation matrices, and heatmaps, which reveal how variables are related to each other.
2. Describing Data Drilling Concepts
Data drilling involves examining data at different levels of granularity. Granularity refers to the level of detail or specificity in the data. EDA explores data at various granularities to understand patterns and trends. For example, drilling down from yearly data to monthly or daily data provides a more detailed perspective.
3. Describing Data Mining Concepts:
Correlation Analysis
EDA examines correlations between variables to identify patterns and relationships. Correlation matrices and scatter plots are commonly used to visualize correlations.



Anomalies
Anomalies are data or values that deviate from the expected pattern in a dataset.
They may include nonsensical values, such as negative ages or fractional quantities, which are unrealistic in real-world scenarios, such as negative age (-15 years), which are implausible in real-world contexts.
Anomalies can be detected through exploratory data analysis (EDA) techniques, such as visualizations like box plots and histograms.
Outliers
Outliers are data points that significantly differ from the majority of the dataset. Such as an age of 80 years is not inherently unreasonable in the context of human age, but its value is significantly distant from the general average, making it an outlier.
They represent notable deviations from the dataset's norm and may indicate interesting insights or data errors.
Outliers are identified using EDA methods like box plots, scatter plots, and z-score analysis.
Anomaly is an outlier HOWEVER not all outliers are anomalies.

Outliers
Outliers don't always need to be discarded; they can be reasonable but fall into the outlier category due to their limited quantity.
Anomalies
Anomalies must be removed as they deviate significantly from the expected pattern or are logically implausible.

Finding first quantile and third quantile

Find the IQR which is the difference between third and first quartile

Find lower and upper bound

Handling Outlier




























KAMIS
Hypothesis Testing
A statistical hypothesis test is a procedure in statistical analysis aimed at determining whether the available data provides enough evidence to support a specific hypothesis. This process usually entails computing a test statistic based on the data. Subsequently, a decision is made by comparing the test statistic to a critical value or by assessing a p-value derived from the test statistic.
T-test
The t-test assesses the difference in means between two groups of data. It's a hypothesis test conducted using random samples from each group. Through this test, analysts determine if the same treatment yields consistent results in both groups or if there are differences.
Accepted hypotheses are:
Ho: No difference between the groups.
Ha: Difference exists despite the same treatment
Z-test
The z-test compares means or proportions between two groups when the sample size is large (typically n > 30) and the population standard deviation is known. It's akin to the t-test but relies on the standard normal distribution (Z-distribution). Commonly used for hypothesis testing when the population standard deviation is known.
Accepted hypotheses are:
Ho: There is no significant difference between the groups.
Ha: A significant difference exists despite the same treatment.
Import Python Libraries
An experiment on the `effects of anti-anxiety medicine on memory recall when being primed with happy or sad memories`. The participants were done on novel Islanders whom mimic real-life humans in response to external factors.
Drugs of interest (known-as) [Dosage 1, 2, 3]:
A - Alprazolam (Xanax, Long-term) [1mg/3mg/5mg]
T - Triazolam (Halcion, Short-term) [0.25mg/0.5mg/0.75mg]
S- Sugar Tablet (Placebo) [1 tab/2tabs/3tabs]

first_name : First name of Islander
last_name : Last. name of Islander
age : Age of Islander
Happy_Saf_group : Happy or. Sad Memory priming block
Dosage : 1-3 to indicate the level of dosage (low - medium - over recommended daily intake)
Drug : Type of Drug administered to Islander
Mem_Score_Before : Seconds - how long it took to finish a memory test before drug exposure
Mem_Score_After : Seconds - how long it took to finish a memory test after addiction achieved
Diff : Seconds - difference between memory score before and after

T-test

At a significance level of 5%, drug exposure has had a significant impact on the time required to complete the memory test.
JUMAT
Machine Learning Regression - Simple Linear Regression

Simple Linear Regression is a statistical method used to model the relationship between a single independent variabel and a dependent variable by fitting a linar equation to the observed data.

Study Case

We use “Salary_Data” dataset to involves predicting about the salary or wages of workers or individuals based on experience



Load and Check Data

Import Library


Load Dataset


Check Data Condition


Data is Clean

Explanatory Data Analysis
Distribution of YearsExperience


Distribution of Salary



Simple Linear Regression
Processing Modeling



Splitting, Training & Test Set


Fitting Into Training


Plot The Result “Bar”

Evaluate Model
Mean Squared Error (MSE)
is a regression model evaluation metric that is used to calculate the error between the values predicted by the model and the actual values. MSE is the average of the squared differences between the predicted value and the actual value.
Mean Absoulute Error (MAE)
is a regression model evaluation metric that is used to calculate the error between the values predicted by the model and the actual values.
R-Squared (R2)
R-squared or R² is one of the regression model evaluation metrics used to calculate the level of success of the model in describing the variance of the target variable.
