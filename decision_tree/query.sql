select
  age,
  income,
  dependents,
  has_property,
  has_car,
  credit_score,
  job_tenure,
  has_education,
  loan_amount,
  date_diff(d, loan_start, loan_deadline) as loan_period,
  greatest(0, date_diff(d, loan_deadline, loan_payed)) as delay_days
from
  default.loan_delay_days