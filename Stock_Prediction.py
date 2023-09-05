import streamlit as st 
import yfinance as yf 
from datetime import date 
from plotly import graph_objs as go 
from prophet import Prophet
from prophet.plot import plot_plotly
START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title('Stock market prediction')

stocks = ('AAPL','GOOG','MSFT','GME')

selected_stock = st.selectbox('Select data',stocks)

n_years = st.slider('Years of prediction', 1 , 4)

period = n_years * 365

@st.cache_data() # decorater ---> caches data (stores) avoids running same function below for same data set
def load_data(ticker):
     # stock name
     data = yf.download(ticker,START,TODAY) # returns in pandas df
     data.reset_index(inplace = True)
     return data

state = st.text('Load data....')
data = load_data(selected_stock)
state.text('Loading...complete.')

st.subheader('Raw data')

st.write(data.tail()) # uses pandas to create raw table with last (k) rows 

def plot_raw():
     fig = go.Figure()
     fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'Stock_Open')) # Takes from header row and column data and plots using plotly and pandas 
     fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'Stock_Close'))
     fig.layout.update(title_text = 'Time series data', xaxis_rangeslider_visible = True) # 2nd argument creates a slider on the x axis to focus on years/months
     st.plotly_chart(fig)

plot_raw() #plots whats in the function which is the specific selected stock 
    
# Forecasting 

df_train = data[['Date','Close']]
df_train = df_train.rename(columns = {'Date':'ds','Close':'y'}) # rename columns for fbprophet format 

model = Prophet() 
model.fit(df_train) # fit training data
future = model.make_future_dataframe(periods = period) # this is the reason for the slider to specify what years to predict using the model
forecast = model.predict(future) # returns forecasts which is a dataframe 



st.subheader('Forecast data')
st.write(forecast.tail())  # Creates a table of raw data (everything below is from this table)

st.write('Forecast graph')
figure_1 = plot_plotly(model,forecast) # plots forecast 
st.plotly_chart(figure_1)

st.write('Forecast componenets')
figure_2 = model.plot_components(forecast) # plots forceast components
st.write(figure_2)
