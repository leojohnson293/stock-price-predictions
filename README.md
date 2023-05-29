# ***Stock Price Prediciton***
> This project utilizes machine learning to predict stocks. The ML techniques used here, are Linear Regression with Regularization and Neutral Networks known as LTSM. This project was done on the Google Collab Python notebook.
## Project Summary
Machine learning and Artifical Intelligence can be used in finance to make predictions on future stock prices. By being able to make accurate predictions on stock, investors can maximize their return and know when to buy/sell securities. The datasets used to train the ML/AI models will be the historical stock prices stored in the 'stock.csv' file and the volume of transactions in the 'stock_volume.csv' file.

The project will uses the Pandas library to create dataframes from the csv file. It will also use the Plotly, Matplotlib and Seaborn libraries to perform data visualization. Finally for the AI/ML modeling, the project will employ Sklearn and Tensorflow to obtain the predicitons.

## Normalization and Visualization
Firstly the stock prices and volume data must be normalised for it to be used in the model. The function below is created to take a Pandas dataframe as an arguement to then, normalise by dividing each row by the first row. This will result in all stocks from starting from a value of one. This will showcase how much the price and volume of each stock has grown relative to the others.
```python
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x
```
For the visualization for each dataset, the following function will take a dataframe and the title for the plot as arguements to create an interactive graph using the Plotly library.
```python
def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()
```

## Linear Regression
### _Simple Linear Regression_
This uses the standard maths formula for a straight line which is 'y = mx + c' to make predictions between two variables.  Y is the dependant variable and x is the independant variable. Once gradient 'm' and intercept 'c' are obtained, the model can use this equation to predict the value of 'y' based on the value of 'x'.     

The least sum of squares method is used to find the gradient and the intercept. This method takes the sum of all the squares of the offsets (the vertical distance of each point on the graph from the line) to estimate the best fit curve or line. The sum of offsets are squared to remove any negative sign for each offset. 
### _Regularization_
 Regularization is used to avoid overfitting which occurs when the ML models take the data points and fail to generalize. One regularization techinque used is Ridge Regression which includes a penalizing term to minimise overfitting. A way to help the model generalize, is to change the slope of the line, so it goes through more data points. The penalizing term is the amount the slop is changed by which is defined as alpha added to the sqaure of the slope. 

The alpha variable used, can be defined in the code below is the `Ridge` method from SkLearn:
```python
from sklearn.linear_model import Ridge

regression_model = Ridge(alpha = 2.0)
```
## LSTM

LSTM (Long short-term memory) is an neutral network that is used in ML models. LSTM avoids the vanishing gradient problem that occurs in a vanilla RNN(recurring neutral network). The vanishing gradient is a problem that occurs when the gradient becomes very small and the predictions become harder to make.

LSTM networks are designed to remember long term dependencies and recall the information for a long period of time. They cantain gates that allow or block information from passing by.

The following code shows the LSTM network used to predict the stock prices for the S&P500. This uses the Keras neutral network library that runs on top of the Tensorflow library.

```python
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences= True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()
```