# This is a sample Python script.

import pandas as pd
import matplotlib.pyplot as mpl
import scipy.stats as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

df = pd.read_csv("cig_data.csv")

print("This study examines the relationship between Carbon Monoxide and the tar content, mass, and nicotine concentration in cigarettes")
print(df)

print()
print("To become better acquainted with some of the popular Python libraries for data science / ML, regressions will be conducted using scikit-learn, scipy, and statsmodels")

print()
print("It is often helpful to first plot the variables so as to get a visual idea of how they're related")

mpl.scatter(df.iloc[:, 2], df.iloc[:, 5])
mpl.title("Tar vs Carbon Monoxide Content")
mpl.xlabel("Tar (mg)")
mpl.ylabel("CO (mg)")
mpl.show()

mpl.scatter(df.iloc[:, 3], df.iloc[:, 5])
mpl.title("Nicotine vs Carbon Monoxide Content")
mpl.xlabel("Nicotine (mg)")
mpl.ylabel("CO (mg)")
mpl.show()

mpl.scatter(df.iloc[:, 4], df.iloc[:, 5])
mpl.title("Cigarette Mass vs Carbon Monoxide Content")
mpl.xlabel("Cigarette Mass (g)")
mpl.ylabel("CO (mg")
mpl.show()

print("As you can see, tar and nicotine concentration seem to have a positive, linear relationship with CO")
print("Plotting mass against CO seems to show no discernable pattern")
print("We now perform a regression analysis using scikit-learn library to rigorously examine the relationships between the variables and attempt to construct a best fit model")

co = df.iloc[:, 5].values
tar = df.iloc[:, 2].values
nicotine = df.iloc[:, 3].values
mass = df.iloc[:, 4].values

tar_train, tar_test, co_train, co_test = train_test_split(tar, co, test_size=0.2, random_state=0)
model = LinearRegression()
#why do we need to use reshape(1, -1) here?
#without it we get the error:
#ValueError: Expected 2D array, got 1D array instead:
model.fit(tar_train.reshape(-1, 1), co_train)
print("Tar Model: ")
print("intercept: " + str(model.intercept_))
print("coefficient: " + str(model.coef_))

nicotine_train, nicotine_test, co_train, co_test = train_test_split(nicotine, co, test_size=0.2, random_state=0)
model.fit(nicotine_train.reshape(-1, 1), co_train)
print("Nicotine Model: ")
print("intercept: " + str(model.intercept_))
print("coefficient: " + str(model.coef_))

mass_train, mass_test, co_train, co_test = train_test_split(mass, co, test_size=0.2, random_state=0)
model.fit(mass_train.reshape(-1, 1), co_train)
print("Mass Model: ")
print("intercept: " + str(model.intercept_))
print("coefficient: " + str(model.coef_))

print()
print("We now carry out a regression using scipy")
print("Analysis for tar: ")
slope, intercept, r_value, p_value, std_err = sp.linregress(df.iloc[:, 2], df.iloc[:, 5])
print("slope: " + str(slope) + ", intercept: " + str(intercept) + " r_value" + str(r_value) + ", p_value " + str(p_value) + " std_err " + str(std_err))

print()
print("Analysis for nicotine: ")
slope, intercept, r_value, p_value, std_err = sp.linregress(df.iloc[:, 3], df.iloc[:, 5])
print("slope: " + str(slope) + ", intercept: " + str(intercept) + " r_value" + str(r_value) + ", p_value " + str(p_value) + " std_err " + str(std_err))

print()
print("Analysis for mass: ")
slope, intercept, r_value, p_value, std_err = sp.linregress(df.iloc[:, 4], df.iloc[:, 5])
print("slope: " + str(slope) + ", intercept: " + str(intercept) + " r_value" + str(r_value) + ", p_value " + str(p_value) + " std_err " + str(std_err))

print()
print("Our analysis confirms that both tar and nicotine content are good predictors of CO concentration, while mass of a cigarette is not")

print()
print("Since tar and nicotine are good linear predictors, would it be beneficial to include both in one model? A multiple regression analysis can reveal the predictive utility of including both in one equation")


print("We now carry out the regression using statsmodels")

#why is this line necessary?
Xtar = sm.add_constant(tar)
reg_model = sm.OLS(co, Xtar).fit()
predictions = reg_model.predict(Xtar)
print_model = reg_model.summary()
print(print_model)

Xnic = sm.add_constant(nicotine)
reg_model = sm.OLS(co, Xnic).fit()
predictions = reg_model.predict(Xnic)
print_model = reg_model.summary()
print(print_model)

Xmas = sm.add_constant(mass)
reg_model = sm.OLS(co, Xmas).fit()
predictions = reg_model.predict(Xmas)
print_model = reg_model.summary()
print(print_model)

X = df[['tar', 'nicotine']]
X = sm.add_constant(X)
reg_model = sm.OLS(co, X).fit()
predictions = reg_model.predict(X)
print_model = reg_model.summary()
print(print_model)