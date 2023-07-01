import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

st.title('S&P 500 App')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, yfinance
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
#
@st.cache_data
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0) # read_html() 자체가 html 내에 있는 table 속성에 해당하는 값만 불러옴
    # st.write(html) # 완전한 dataframe 형식
    df = html[0]
    return df

df = load_data()
sector = df.groupby('GICS Sector')
# st.write(sector.first()) # GICS Sector로 정렬, 각 sector의 첫번째 record를 보여주는 거지...
# st.write(sector.describe())
# st.write(sector.get_group('Health Care'))

# Sidebar - Sector selection
sorted_sector_unique = sorted( df['GICS Sector'].unique() ) 
# st.write(len(sorted_sector_unique)) # 11개
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Filtering data
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)

# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

data = yf.download(
        tickers = list(df_selected_sector[:10].Symbol),
        period = "ytd", # year to date
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )

# st.write(df.Symbol)
# st.write(data)

# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot(plt)

num_company = st.sidebar.slider('Number of Companies', 1, 5)

if st.button('Show Plots'): # plot할 symbol을 선택하는 부분이 구현되지 않음!
    st.header('Stock Closing Price')
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)

# 대충 이렇게 구현할 수 있을 거 같은데...
# df에 없는 symbol까지 다 나와버리니까...
# sorted_symbol_unique = sorted(df['Symbol'].unique() ) 
# option = st.selectbox('choose', sorted_symbol_unique)
    
# if st.button('show plots'):
#     price_plot(option)
