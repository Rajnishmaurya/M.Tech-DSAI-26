import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from math import log
from sklearn import linear_model

data=pd.read_csv("/home/rajnish/Desktop/IIT MADRAS M.Tech DSAI 2026/DAL(DATA ANALYTICS LABORATORY)/Assignments/Assignments2/Assignment2.data",sep='\t')
data

data.describe()


fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Two Cyber Physical Systems')
fig.set_figwidth(10)
fig.set_figheight(5)

ax1.plot(data.SpringPos, 'r+')
ax1.set_ylabel('Spring Position')
ax2.plot(data.StockPrice, 'b.')
ax2.set_ylabel('Stock Price')
ax2.set_xlabel('time')


plt.show()


# # Spring Positon Dataset

# # Task 1

# 1. Implement the OLS closed form solution using numpy’s matrix operators to find the value of ‘m’ that minimizes SSE.


y2 = pd.DataFrame({"x":range(226), "y":data.SpringPos})

y2.head



yy=np.array(y2.y)
xx=np.expand_dims(y2.x,axis=1)



numerator = np.matmul(np.transpose(xx), yy)
denom = np.matmul(np.transpose(xx), xx)
denom_inv = np.linalg.inv(denom)
beta = np.matmul(denom_inv, numerator)
print("Beta = ", beta[0])
sse = np.sum((xx*beta[0] - np.expand_dims(yy,1))**2)
print("SSE = ", sse)



# estimate the value of the beta vector assuming that X is made of independent features.
def estimateBeta(X, y):
    numerator = np.matmul(np.transpose(X), y)
    denom = np.matmul(np.transpose(X), X)
    denom_inv = np.linalg.inv(denom)
    beta = np.matmul(denom_inv, numerator)
    return beta

# create a helper that would estimate yhat from X and beta.
def predict(beta, X):
    # reshape the input to a matrix, if it is appearing like an 1d array.
    if len(X.shape) != 2:
        X = np.expand_dims(X,1)
    # convert the beta list in to an array.
    beta = np.array(beta)
    # perform estimation of yhat.
    return np.matmul(X, beta)

# compute the sum of squared error between y and yhat.
def SSE(y, yhat):
    return np.sum((y-yhat)**2)



plt.plot(y2.x, y2.y, 'r+')
yhat1 = predict(beta, y2.x)
plt.plot(y2.x, yhat1, 'b-')  # yhat = y2.x*beta[0]
plt.ylabel('SpringPos')
plt.xlabel('Time')



y2df = pd.DataFrame({"bias":np.ones(226), "x":range(226), "y":data.SpringPos})
yy = np.array(y2df.y) 
xx = np.array(y2df[["bias","x"]])
y2df.head()



beta2 = estimateBeta(xx, yy)
print("beta =", beta2)
yhat2 = predict(beta2, xx)
loss = SSE(yy, yhat2)
print("SSE =", loss)



plt.plot(y2df.x, y2df.y, 'r+')
plt.plot(y2df.x, yhat1, 'b-')
plt.plot(y2df.x, yhat2, 'g-')
plt.ylabel('SpringPos')
plt.xlabel('Time')



x1 = round(y2.x*beta2[1],2)
x2 = np.sin(x1)

y21 = pd.DataFrame({"bias":np.ones(226),"x":range(226), "x1":x1, "x2":x2, "y":data.SpringPos})
y21.head(10)




xx = np.array(y21[['bias', 'x1', 'x2']])
yy = np.array(y2.y) 




beta3 = estimateBeta(xx, yy)
print("Beta = ", beta3)
yhat3 = predict(beta3, xx)
loss = SSE(yy, yhat3) #np.sum((np.matmul(xx,beta) - yy)**2)
print("SSE = ", loss)



plt.plot(y2.x, y2.y, 'r+')
plt.plot(y2.x, yhat3, 'b-')
plt.ylabel('Stock Price')
plt.xlabel('Time')




print("Best best=",beta3)
print("SSE=",loss)


# 2. Implement a linear search (the single parameter search version of grid search) for m = tan θ,where θ in [0, 60] in steps of 5 degrees and measure the SSE at every choice of θ. Create a plot that shows SSE vs θ. Report the θ, that minimizes SSE.
# 
# 


data



import numpy as np
import matplotlib.pyplot as plt
y2 = pd.DataFrame({"x":range(226), "y":data.SpringPos})

y2.head



# Define the SSE function
def compute_sse(m, x, y):
    # SSE = Sum of Squared Errors
    y_pred = m * x
    sse = np.sum((y - y_pred) ** 2)
    return sse

# Generate range of theta values from 0 to 60 degrees in steps of 5 degrees
theta_values = np.arange(0, 65, 5)
sse_values = []

# Linear search through theta values
for theta in theta_values:
    m = np.tan(np.radians(theta))  # Convert theta to radians and compute m = tan(theta)
    sse = compute_sse(m, y2.x, y2.y)
    sse_values.append(sse)

# Find the theta that minimizes SSE
min_sse = min(sse_values)
min_theta = theta_values[sse_values.index(min_sse)]

# Plot SSE vs. theta
plt.plot(theta_values, sse_values, marker='o')
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel('SSE')
plt.title('SSE vs. θ')
plt.grid(True)
plt.show()

# Report the theta that minimizes SSE
min_theta, min_sse



print("Degree=",min_theta)
print("Min SSE=",min_sse)


# 3. Implement the solution using sklearn’s LinearRegression class.


x1 = round(y2.x*beta2[1],2)

y21 = pd.DataFrame({"bias":np.ones(226),"x":range(226), "x1":x1, "y":data.SpringPos})
y21.head(10)



xx = np.array(y21[['x1']])
yy = np.array(y2.y) 



from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree = 13)   # 10, 11
X_poly = poly_transformer.fit_transform(xx)



from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_poly, yy)
print("Intercept=", model.intercept_, "Beta = ", model.coef_)
yhat4 = model.predict(X_poly)

#beta4 = estimateBeta(X_poly, yy)
#print("Beta = ", beta4)
#yhat4 = predict(beta4, X_poly)

loss = SSE(yy, yhat4)
print("SSE = ", loss)



plt.plot(y2.x, y2.y, 'r+')
plt.plot(y2.x, yhat4, 'b-')
plt.ylabel('SpringPos')
plt.xlabel('Time')
plt.legend(['Price', 'Linear Regression'], loc='upper left')
plt.show()



plt.ylabel('Spring Position')
plt.xlabel('Time')
plt.plot(y2.x, y2.y, 'r+')
plt.plot(y2.x, yhat2, 'g-')
plt.plot(y2.x, yhat3, 'b-')
plt.plot(y2.x, yhat4, 'y-')

# plt.plot(y2.x, yhat2, '--')
# plt.plot(y2.x, yhat3, 'k')
plt.legend(['Spring Position', 'OLS','Line+Sinusoid','Linear Regression'], loc='upper left')
plt.show()


# # Task 2 (Spring Position Dataset)

# You will notice that the linear model is an ok fit for the y2. What should be the mathematical model of stock price dataset? If you notice the periodicity in the data, you should factor that in your mathematical model using an appropriate function that’s periodic. The challenge here is; the trend of the magnitude is also increasing, which you confirmed in your previous task. So, the math model should consider both properties.

# 1. Split your data into Train, Eval & Test.
#    #i. Interpolation: When you randomly split the data into train, eval and test; your test and evaluation data points may be inside the data range (time range). When you can predict those points correctly, you are essentially recovering missing data in the regression line.This is also called the interpolation problem.
#    ii. Extrapolation: In this scenario, the test and eval points should be outside the time range of the training data. If your model is a good fit, and when you predict the data point outside the range, you are essentially extrapolating the regression line. This is also called the “Forecasting” task.
#    


data



df = pd.DataFrame({"bias":np.ones(226), "x":range(226), "y":data.SpringPos})



amplitude = (df[['y']].max() - df[['y']].min()) / 2
b = 0.005
w = 2 * np.pi / 62
x_wave = np.exp(-1*b*df[['x']])*np.sin(w*df[['x']]) @ [amplitude] 
x_wave



df = pd.DataFrame({"bias":np.ones(226), "x":range(226)})
df["x_wave"]=x_wave
df["y"]=data.SpringPos
df




yy = np.array(df.y) 
xx = np.array(df[["bias","x","x_wave"]])




# from sklearn.model_selection import train_test_split
# import numpy as np
# X_train,X_test,Y_train,Y_test=train_test_split(xx, yy, test_size=0.2, random_state=42)





X_train=xx[0:150]
Y_train=yy[0:150]

X_test=xx[151:190]
Y_test=yy[151:190]

X_val=xx[191:226]
Y_val=yy[191:226]




# X_train,X_val,Y_train,Y_val=train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


# 2. Implement the regression model (OLS or LinearRegression or equivalent) using appropriate feature transformation so that the SSE is lower than that of Task 1.



# estimate the value of the beta vector assuming that X is made of independent features.
def estimateBeta(X, y):
    numerator = np.matmul(np.transpose(X), y)
    denom = np.matmul(np.transpose(X), X)
    denom_inv = np.linalg.inv(denom)
    beta = np.matmul(denom_inv, numerator)
    return beta

# create a helper that would estimate yhat from X and beta.
def predict(beta, X):
    # reshape the input to a matrix, if it is appearing like an 1d array.
    if len(X.shape) != 2:
        X = np.expand_dims(X,1)
    # convert the beta list in to an array.
    beta = np.array(beta)
    # perform estimation of yhat.
    return np.matmul(X, beta)

# compute the sum of squared error between y and yhat.
def SSE(y, yhat):
    return np.sum((y-yhat)**2)




beta2 = estimateBeta(X_train, Y_train)
print("beta =", beta2)
yhat2 = predict(beta2, xx)
loss = SSE(yy, yhat2)
print("SSE =", loss)


# 3. Train the regression model for interpolation and evaluate the SSE.



beta2 = estimateBeta(X_train, Y_train)
print("beta =", beta2)
yhat2 = predict(beta2, X_test)
loss = SSE(Y_test, yhat2)
print("SSE =", loss)


# 4. Train the regression model for extrapolation and evaluate the SSE.




beta2 = estimateBeta(X_train, Y_train)
print("beta =", beta2)
yhat2 = predict(beta2, X_val)
loss = SSE(Y_val, yhat2)
print("SSE =", loss)
















