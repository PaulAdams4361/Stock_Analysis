from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import time
import os
import os.path
import pathlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy.fftpack import fft, fftshift
from datetime import datetime
import glob
import altair as alt
import altair_saver
from pandas import json_normalize
import requests
        
class AlphaVantage_SMA:

##############################################
##############################################
# Getting newest symbols available on NASDAQ #
##############################################
##############################################

    def __init__(self, dirloc=None, time_period=7):
        '''
        This script sets up the API key and directories to store data
        dirloc should be a string like: '/home/pablo/Desktop/'
        '''
        self.time_period = time_period
        self.dirlocation = dirloc + '/NASDAQ_{}'.format(datetime.date(datetime.now())) # YYYY_MM_DD
        
        if os.path.exists(self.dirlocation):
            os.chdir(self.dirlocation)
        else:
            path = pathlib.Path(self.dirlocation)
            path.mkdir(parents=True, exist_ok=True)
            os.chdir(self.dirlocation)
            
        make_dirs = ['./STOCKS_DAILY','./SMA_DAILY','./ANALYSIS','./DAILY_CHARTS_SPECTRAL'
                     ,'./DAILY_SMA_COMPARE_CHARTS','DAILY_EXPLOSIVE_REGRESSION_CHARTS']
        
        for make in make_dirs:
            if os.path.exists(make):
                pass
            else:
                os.mkdir(make)
                
            
        self.api_key = ''
        
    def get_nasdaq(self):
        path = self.dirlocation
        os.chdir(path)
        #os.chdir(self.path)
        os.system("curl --ftp-ssl anonymous:jupi@jupi.com "
                  "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt "
                  "> nasdaq.lst")

        # create the (mostly) special-character removed .lst2:
        os.system("tail -n +9 nasdaq.lst | cat | sed '$d' | sed 's/|/ /g' > " "nasdaq.lst2")

        # convert the .lst2 to the nasdaq.csv file
        os.system("awk '{print $1}' nasdaq.lst2 > nasdaq.csv")
        os.system("echo; head nasdaq.csv; echo '...'; tail nasdaq.csv")

        # You may need to manually enter the csv and remove the junk headings in order for pandas to be able to read the csv:
        symbolsNASDAQ = pd.read_csv('./nasdaq.csv', usecols=[0])
        symbols = symbolsNASDAQ.iloc[34:,0].reset_index(drop=True).tolist()
        symbols_to_write = pd.DataFrame(symbols, columns = ['symbols'])
        symbols_to_write.to_csv('./nasdaq.csv', index=False)
        
        return symbols
                    
    def get_daily_sma(self):
        
        path = self.dirlocation
        os.chdir(path)
        
        symbols = AlphaVantage_SMA.get_nasdaq(self)
        time_period = self.time_period
        #symbols = ['AAXN', 'ALXN']
        api_key = self.api_key
        
        ts = TimeSeries(api_key, output_format='pandas')

        ######## SMA config
        TECHNICAL_ANALYSIS_SMA_KEY = 'Technical Analysis: SMA'
        SMA_KEY = 'SMA'
        
        def simple_moving_average_daily(symbols, time_period): # right now, time_period = 7 for a 7-day moving average for a few weeks' of trading
        #def simple_moving_average_daily(symbols): # right now, time_period = 7 for a 7-day moving average for a few weeks' of trading
            api_url = 'https://www.alphavantage.co/query?function=SMA&symbol={}&interval=daily&time_period=7&series_type=high&apikey='.format(symbols)
            results = []
            response = requests.get(api_url)
            if response.status_code == requests.codes.ok:
                api_result = response.json()
                if TECHNICAL_ANALYSIS_SMA_KEY in api_result:
                    sma_dict = api_result[TECHNICAL_ANALYSIS_SMA_KEY]
                    for date_key in sorted(sma_dict):
                        sma_value = sma_dict[date_key][SMA_KEY]
                        observation = {
                            'date': date_key,
                            'SMA': float(sma_value)
                        }
                        results.append(observation)

            return results
        ######## end SMA config
        # Read the STOCKS_DAILY files, download SMA data, horizontall stack SMA to the right of STOCKS_DAILY
        errFile = open("badSymbols.txt", "w+")
        #os.chdir('~/Desktop/')
        for i in range(0, len(symbols)):
            ti = TechIndicators(symbols[i])

            try:
                stock_daily, stock_meta_data_daily = ts.get_daily(symbol=symbols[i], outputsize='compact')

                #fileOut = ("./STOCKS_DAILY/NASDAQ_Daily_" + symbols[i] + ".csv")
                fileOut = (path+"/STOCKS_DAILY/NASDAQ_Daily_{}.csv".format(symbols[i]))
                fileOut = str(fileOut)
                stock_daily['Symbol'] = symbols[i]
                stock_daily.reset_index(level=0, inplace=True)
                stock_daily.rename(columns = {'1. open': 'open_price', '2. high': 'high_price','3. low': 'low_price', '4. close': 'close_price','5. volume': 'trade_volume'}, inplace=True)

                sma_data = simple_moving_average_daily(symbols[i], time_period=time_period)
                sma_df = json_normalize(sma_data).sort_values(by='date', ascending=False).reset_index(drop=True)
                sma_df = sma_df.head(len(stock_daily)).copy() # truncate moving averages to compare to the length of available stock prices

                daily_df = pd.concat([stock_daily, sma_df['SMA']], axis=1).reset_index(drop=True)
                daily_df.to_csv(fileOut, index=False)

                time.sleep(3)

            except:
                with open("badSymbols.txt", "a") as myfile:
                    myfile.write(symbols[i] + "\n")

    def analyze_sma(self):

        if len(files) == 0:
            AlphaVantage_SMA.get_daily_sma(self)
        else:
            pass
        
        path = self.dirlocation
        os.chdir(path)
        
        files = os.listdir('./STOCKS_DAILY') # list all the files in the directory

        for file in files:
            df = pd.read_csv(f'./STOCKS_DAILY/' + file) # read each files in the directory and for each file:

            try:
                if(df.shape[0] > 15): # if there are more than 15 records
                    x = np.array(list(range(df.shape[0]))) + 1 # create a time variable equal to the length of the file plus 1 (since indexing starts at 0)
                    x = x.copy().reshape((-1, 1)) # Re-shape to create the vector space needed for regression (since regression takes place on an x-y 2D planar graph)
                    y = np.array(df['low_price'].iloc[::-1]) # the target variable will be the low price

                    model = LinearRegression().fit(x, y) # run the regression
                    model.score(x, y) # gather statistics on the regression

                    # model.coef_ = slope; model.intercept_ = current price; df['trade_volume'][0] = most recent volume:
                    #if ((model.coef_ > 0.04)
                    if ((model.coef_.item() > 0.2)
                        #& (15.00 <= model.intercept_<= 110.00) 
                        & (df['trade_volume'].iloc[1]> 1000)): # positive slope > 0.04 (profitable), price between $15-110, volume > 1000 (liquidity)

                        df['sma_diff'] = df['low_price'] - df['SMA'] # subtract the high moving average from the low price
                        # if each of the most recent 2 days have low prices greater than the moving average
                        # and the average of the most recent 5 days have a positive difference of low price minus sma
                        #if (df.iloc[0,8] > 0) & (df.iloc[1,8] > 0) & (df.iloc[0:4,8].mean() > 0):
                        if (df.iloc[0,8] > 0) & (df.iloc[1,8] > 0):
                        #if (df.iloc[0,8] > 0):
                            with open("./ANALYSIS/stock_picks_daily_sma.txt", "a+") as f:
                                f.write(df['Symbol'][0]) # write the unique symbol to file if all the above investing criteria is met
                                f.write("\n")

            except:
                pass

        df_daily_sma_diff = pd.read_csv('./ANALYSIS/stock_picks_daily_sma.txt', header=None)
        unique_stocks_daily_sma = pd.Series(df_daily_sma_diff[0].unique())
        unique_stocks_daily_sma.to_csv('./ANALYSIS/stock_picks_daily_sma.csv')

    def get_sma(self):
        try:
            unique_stocks_daily_sma = AlphaVantage_SMA.analyze_sma(self)

            path = self.dirlocation
            os.chdir(path)

            for i in np.arange(len(unique_stocks_daily_sma)):
                symbol = unique_stocks_daily_sma[i]
                path = './STOCKS_DAILY/NASDAQ_Daily_{}.csv'.format(symbol) #using symbol so it's easier to follow
                stock = pd.read_csv(path)

                chart = alt.Chart(stock).mark_line(strokeWidth=4).encode(
                    alt.X('date:T',axis=alt.Axis(title='Month', labelAngle=0)),
                    alt.Y('low_price:Q',axis=alt.Axis(title='Price', labelAngle=0)),
                    tooltip=['low_price', 'trade_volume']
                ).properties(title='{} (NASDAQ) Daily Prices, FY 2016 - Present'.format(stock['Symbol'][1]), width=1000, height=250).configure_axis(
                    labelFontSize=14,
                    titleFontSize=18).configure_title(
                    fontSize=22, anchor='start')

                chart.save('./DAILY_SMA_COMPARE_CHARTS/{}_chart.html'.format(stock['Symbol'][1]), scale_factor=2.0)


                open_close_color = alt.condition("datum.open_price <= datum.close_price", alt.value("#06982d"), alt.value("#ae1325"))

                base = alt.Chart(stock).encode(
                    alt.X('date:T',
                          axis=alt.Axis(
                              format='%m/%d',
                              labelAngle=-45,
                              title='Date in 2009'
                          )
                    ),
                    color=open_close_color
                )

                rule = base.mark_rule().encode(
                    alt.Y(
                        'low_price:Q',
                        title='Price',
                        scale=alt.Scale(zero=False),
                    ),
                    alt.Y2('high_price:Q'),
                        tooltip=['low_price', 'trade_volume']
                ).properties(width=800,
                    height=300)

                bar = base.mark_bar().encode(
                    alt.Y('open_price:Q'),
                    alt.Y2('close_price:Q')
                )

                charted = rule + bar
                charted.save('./DAILY_SMA_COMPARE_CHARTS/{}_candlestick_chart.html'.format(stock['Symbol'][1]))
        except:
            pass
        
        
##################################################################
##################################################################
##################################################################

    def analyze_explosive(self):
        '''
        This function uses linear regression on the last 5% of the data to check if there was a high growth rate
        and if there was a positive growth rate over the last 15% of the datapoints.
        '''
        
        path = self.dirlocation
        os.chdir(path)
        
        files = os.listdir('./STOCKS_DAILY') # list all the files in the directory
        
        ################################################
        ### To check if the stock data is already there:
        ################################################
        
        if len(files) == 0:
            AlphaVantage_SMA.get_daily_sma(self)
        else:
            pass

        for file in files:
            df = pd.read_csv(f'./STOCKS_DAILY/' + file) # read each files in the directory and for each file:

            if (df.shape[0] > 15): # if there are more than 15 records
                x = np.array(list(range(df.shape[0]))) + 1 # create a time variable equal to the length of the file plus 1 (since indexing starts at 0)
                x = x.copy().reshape((-1, 1)) # Re-shape to create the vector space needed for regression (since regression takes place on an x-y 2D planar graph)
                y = np.array(df['low_price'].iloc[::-1]) # the target variable will be the low price

                model = LinearRegression().fit(x, y) # run the regression
                model.score(x, y) # gather statistics on the regression

                # model.coef_ = slope; model.intercept_ = current price; df['trade_volume'][0] = most recent volume:
                #if ((model.coef_ > 0.04)
                if ((model.coef_.item() > 0.04) & (model.coef_.item() < 0.2)
                    #& (15.00 <= model.intercept_<= 110.00) 
                    & (df['trade_volume'].iloc[1]> 1000)): # positive slope > 0.04 (profitable), price between $15-110, volume > 1000 (liquidity)

                    df_15_pct = df.iloc[:(np.floor(len(df)*0.1).astype(int)),:] ##### EXPLOSIVE if 0.05 or less, otherwise, 15% or more.

                    x = np.array(list(range(df_15_pct.shape[0]))) + 1
                    x = x.copy().reshape((-1, 1))
                    y = np.array(df_15_pct['low_price'].iloc[::-1])

                    model_15_pct = LinearRegression().fit(x,y) # run the regression
                    model_15_pct.score(x, y) # gather statistics on the regression

                    #if (model_15_pct.coef_.item() > model.coef_.item()): # if the most recent 15% of the time series has slope > than overall then:
                    if (model_15_pct.coef_.item() > 0.5): # if the most recent 15% of the time series has slope > 0.5 then:
                        with open("./stock_picks_daily_explosive_regression.txt", "a+") as f:
                            f.write(df['Symbol'][0]) # write the unique symbol to file if all the above investing criteria is met
                            f.write("\n")

        df_daily_explosive_regression = pd.read_csv('./stock_picks_daily_explosive_regression.txt', header=None)
        unique_daily_explosive_regression = pd.Series(df_daily_explosive_regression[0].unique())
        unique_daily_explosive_regression.to_csv('./stock_picks_daily_explosive_regression.csv')
        
        return unique_daily_explosive_regression

    def get_explosive(self):
        try:
            unique_daily_explosive_regression = AlphaVantage_SMA.analyze_explosive(self)

            path = self.dirlocation
            os.chdir(path)

                                                            #**************#
                                                            #**************#
                                                            #  Line Plots  #
                                                            #**************#
                                                            #**************#

            for i in np.arange(len(unique_daily_explosive_regression)):
                symbol = unique_daily_explosive_regression[i]
                path = './STOCKS_DAILY/NASDAQ_Daily_{}.csv'.format(symbol)
                stock = pd.read_csv(path)

                chart = alt.Chart(stock).mark_line(strokeWidth=4).encode(
                    alt.X('date:T',axis=alt.Axis(title='Month', labelAngle=0)),
                    alt.Y('low_price:Q',axis=alt.Axis(title='Price', labelAngle=0)),
                    tooltip=['low_price', 'trade_volume']
                ).properties(title='{} (NASDAQ) Daily Prices, FY 2016 - Present'.format(stock['Symbol'][1]), width=1000, height=250).configure_axis(
                    labelFontSize=14,
                    titleFontSize=18).configure_title(
                    fontSize=22, anchor='start')

                chart.save('./DAILY_EXPLOSIVE_REGRESSION_CHARTS/{}_line_chart.html'.format(stock['Symbol'][1]), scale_factor=2.0)



                                                        #**********************#
                                                        #**********************#
                                                        #  Candelsticks Plots  #
                                                        #**********************#
                                                        #**********************#



                open_close_color = alt.condition("datum.open_price <= datum.close_price", alt.value("#06982d"), alt.value("#ae1325"))

                base = alt.Chart(stock).encode(
                    alt.X('date:T',
                          axis=alt.Axis(
                              format='%m/%d',
                              labelAngle=-45,
                              title='Date in 2009'
                          )
                    ),
                    color=open_close_color
                )

                rule = base.mark_rule().encode(
                    alt.Y(
                        'low_price:Q',
                        title='Price',
                        scale=alt.Scale(zero=False),
                    ),
                    alt.Y2('high_price:Q'),
                        tooltip=['low_price', 'trade_volume']
                ).properties(width=800,
                    height=300)

                bar = base.mark_bar().encode(
                    alt.Y('open_price:Q'),
                    alt.Y2('close_price:Q')
                )

                charted = rule + bar
                charted.save('./DAILY_EXPLOSIVE_REGRESSION_CHARTS/{}_candlestick_chart.html'.format(stock['Symbol'][1]))

        except:
            pass
