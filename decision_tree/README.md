# Decision Tree from scratch
This is DecisionTreeRegressor written only using `python` and `numpy`.
The task is to predict how many days the lender will be overdue on the loan.

Data is fethced from ClickHouse.
### Table structure:
| Column        | Descriptions                        |
| ------------- | ----------------------------------- |
| id            | Client id.                          |
| age           | Client age.                         |
| income        | Monthly income.                     |
| dependents    | Number of dependent family members. |
| has_property  | Property ownership.                 |
| has_car       | Car ownership.                      |
| credit_score  | Credit score.                       |
| job_tenure    | Tenure (in years)           .       |
| has_education | Education (0 - no, 1 - yes).        |
| loan_amount   | Loan amount.                        |
| loan_start    | Loan issue date.                    |
| loan_deadline | Scheduled date of full repayment.   |
| loan_payed    | Actual date of full repayment.      |


Data was obtained with a [SQL-query](query.sql).

Features:
- Uses **weighted mse** as base error function.
- Based on **depth wise** growth.
- Has `as_json()` method that outputs tree's structure in JSON format.

[DecisionTree file.](tree.py)

[Notebook](decision_tree.ipynb) with data and predictions.