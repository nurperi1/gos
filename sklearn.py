 import numpy as np
 from sklearn.linear_model import LinearRegression

 #Sample data
 X=np.array([[1], [2], [3], [4], [5]])
 y=np.array([2,3,5,7,11])
 model = LinearRegression()
 model.fit(X,y)

 predictions = model.predict(np.array([[6], [7]]))
 print(predictions) 