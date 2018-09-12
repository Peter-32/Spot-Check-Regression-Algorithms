# Ten Regression

Quickly find a good regression model.

### Steps

- Define the Problem
  - X Define the goal
  - X List the data sources
  - X List the data descriptions
  - Find missing values
  - Find infinity values
  - Identify which column is the target
  - Identify the category columns
- Prepare Data
  - Clean missing and infinity values
  - Drop any category columns
  - Rename the target to y
- Spot Check Algorithms
  - Find the CV code to use
  - Try CV on baseline
- Improve Results
  - Try CV on Linear Regression
  - Try CV on 9 other regression models
  - Function: Run all 10 and identify the best model
  - Function: Change the parameters of that model at least 10 times and identify the best model
  - Output a plot of the results of all models compared to baseline
- Present Results
  - Write an Confluence document:
    - understandable, basic, in isolation, and plots
  - Compare best model with Kaggle results
  - Save the tool as a module

### Data Sources

housingdata: https://www.kaggle.com/apratim87/housingdata

### Data Description

- Data description was copied from: https://www.kaggle.com/apratim87/housingdata

- CRIM per capita crime rate by town ZN proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS proportion of non-retail business acres per town
- CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX nitric oxides concentration (parts per 10 million)
- RM average number of rooms per dwelling
- AGE proportion of owner-occupied units built prior to 1940 DIS weighted distances to five Boston employment centres
- RAD index of accessibility to radial highways
- TAX full-value property-tax rate per $10,000
- PTRATIO pupil-teacher ratio by town
- B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT % lower status of the population
- MEDV Median value of owner-occupied homes in $1000's

### Confluence Document

Link:
