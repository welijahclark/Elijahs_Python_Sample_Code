# importing pandas library
import pandas as pd

# 1.a
list_1a = [5.5, 4.2, 7.0, 0.6, 8.1]
index_1a = ['b', 'c', 'd', 'f', 'g']
series_1a = pd.Series(list_1a, index_1a)
series_1a

# 1.b
series_1b = series_1a
series_1b["z"] = -1.0
print(series_1b)

series_1b2 = series_1b
series_1b2["f"] = 2.6
print(series_1b2)

# 1.c
Cars = pd.read_csv("Car_Ins.csv")
Cars.shape

Cars.head(20)

# 1.d
Cars.info()
d1 = Cars.loc[25, 'Damage']
print(d1)
d2 = Cars.iloc[144:151]
print(d2)

# 1.e
Cars.describe()

Cars.describe(include=object)

pd.crosstab(Cars["Style"], Cars["Size"])

# 1.f
Cars_2 = Cars.dropna()
print(Cars_2)

f2 = Cars_2.iloc[144:151]
print(f2)

# 1.g
med_prop = Cars["Property"].median()
Cars.fillna(med_prop, inplace=True)
print(med_prop)

g2 = Cars.iloc[144:151]
print(g2)

# 1.h
for x in Cars.index:
    if Cars.loc[x, "Collision"] > 250:
        Cars.loc[x, "Collision"] = 160

Cars.loc[51]

# 2.a
a = ('statistically')

for x in a:
    if len(a) > 10:
        print("a is long")
        break
    elif len(a) <= 10 and len(a) >= 5:
        print("a is medium length")
        break
    else:
        print("a is short")
        break

# 2.b
a = ('statistics')

for x in a:
    if len(a) > 10:
        print("a is long")
        break
    elif len(a) <= 10 and len(a) >= 5:
        print("a is medium length")
        break
    else:
        print("a is short")
        break

a = ('stat')

for x in a:
    if len(a) > 10:
        print("a is long")
        break
    elif len(a) <= 10 and len(a) >= 5:
        print("a is medium length")
        break
    else:
        print("a is short")
        break

# 2.c
mod2 = lambda x: x % 2
for n in range(3, 24, 3):
    if mod2(n) == 0:
        print(n, "is even")
    else:
        print(n, "is odd")

# importing numpy
import numpy as np

# 2.d
mod2 = lambda x: x % 2
list_2d = [9, 25, 49, 81, 121]

for x in list_2d:
    output = np.exp(np.log(x))
    print(output)

# reimporting/importing more stuff
import pandas as pd
import scipy.stats as st
import numpy as np

# 3.a
Cars = pd.read_csv("Car_Ins.csv")

xbar = Cars['Collision'].mean()
dof = len(Cars['Collision']) - 1
se = st.sem(Cars['Collision'])

st.t.interval(confidence=.95, df=dof, loc=xbar, scale=se)

# 3.b
st.ttest_1samp(Cars['Collision'], 152, alternative='less')

# reimporting again
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# 4.a
Move = pd.read_csv("Movies_Data.csv")
sns.regplot(x=Move['TrailerViews'], y=Move['BoxOffice'])
plt.show()

# 4.b
x = Move['TrailerViews']
y = Move['BoxOffice']

x1 = sm.add_constant(x)

MoveModel = sm.OLS(y, x1)

results = MoveModel.fit()

print(results.summary(slim=True))

# 4.c
import statsmodels.formula.api as smf

multi_reg = smf.ols('BoxOffice ~ Budget + ScreenCoverage + RunTime + TrailerViews + DirectorRating', data=Move)
mr_results = multi_reg.fit()
mr_results.summary(slim=True)

# 4.d
multi_reg4d = smf.ols('BoxOffice ~ Budget + ScreenCoverage + TrailerViews + DirectorRating', data=Move)
mr_results4d = multi_reg4d.fit()
mr_results4d.summary(slim=True)

# 5.a
X = Move.drop(
    ["BoxOffice", "MovieID", "MarketingCost", "RunTime", "ActorRating", "ActressRating", "ProducerRating", "3DOption",
     "SocMedia", "Genre", "ActorAvgAge", "Screens"], axis=1)
y = Move["BoxOffice"]
print(pd.concat([y, X], axis=1).head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3732)

print(X_train.head(), "\n\n", X_test.head())

# 5.b
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

model5b = DecisionTreeRegressor(random_state=3732)

model5b.fit(X_test, y_test)

pred = model5b.predict(X_test)

print(metrics.r2_score(y_test, pred))
sns.scatterplot(x=y_test, y=pred)

print(pred)

# 5.c
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

model_5c = DecisionTreeRegressor(max_depth=3, min_samples_split=4, random_state=3732)
model_5c.fit(X_train, y_train)

from sklearn.tree import plot_tree

plt.figure(figsize=(16, 8), dpi=180)
plot_tree(model_5c, feature_names=X.columns)
plt.show()

# 5.d
pred2 = model_5c.predict(X_test)

print(metrics.r2_score(y_test, pred2))