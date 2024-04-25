
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 1000000
    n = len(x)
    lr = 0.0002
    old_cost = 0
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - lr * md
        b_curr = b_curr - lr * bd
        if math.isclose(cost, old_cost, rel_tol=1e-20):
            break
        old_cost = cost
        # print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))

    return m_curr, b_curr

def predict_with_sklearn():
    df = pd.read_csv("test_scores.csv")
    reg = LinearRegression()
    reg.fit(df[['math']], df.cs)
    return reg.coef_, reg.intercept_


if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x, y)
    print("Using Gradient Descent Function: Coef {}, Intercept {}".format(m, b))

    m_, b_ = predict_with_sklearn()
    print("Using Sklearn: Coef {}, Intercept {}".format(m_, b_))



