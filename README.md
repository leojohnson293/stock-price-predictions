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
## Training and Testing Data

```python

```
## Linear Regression

```python

```
## LSTM
git init
 add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/leojohnson293/stock-price-predictions.git
git push -u origin main