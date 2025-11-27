import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from lazypredict.Supervised import LazyRegressor



data = pd.read_csv(r"D:\HocPython\NCKH\Vingroup_4y.csv")
# profile = ProfileReport(data,title="Vin Report",explorative=True)
# profile.to_file("Vin_report.html")
# data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# plt.show()

data = data.drop(columns=['time'])

target = "close"
x = data.drop(target,axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
# for i,j in zip(y_predict, y_test):
#     print("Predicted value: {} . Actual value: {}".format(i,j))

print("MAE: {}". format(mean_absolute_error(y_test,y_predict)))
print("MSE: {}". format(mean_squared_error(y_test,y_predict)))
print("R2: {}". format(r2_score(y_test,y_predict)))

# model = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models,prediction = model.fit(x_train,x_test,y_train, y_test)
# print(models)
