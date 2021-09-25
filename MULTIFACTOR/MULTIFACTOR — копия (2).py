import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web
# os.chdir('PERFORMANCE_ANALYSIS FOLDER LOCATION')
import requests
from performance_analysis import annualized_return
from performance_analysis import annualized_standard_deviation
from performance_analysis import max_drawdown
from performance_analysis import gain_to_pain_ratio
from performance_analysis import calmar_ratio
from performance_analysis import sharpe_ratio
from performance_analysis import sortino_ratio
import yfinance as yf
import pprint

import apiclient.discovery
from oauth2client.service_account import ServiceAccountCredentials
import httplib2

import gspread as gd

from yahoo_fin import stock_info as si


# подключение к гугл таблице

LIST = 'Британия'
index = '^FTSE'
KEY = '2105b9f242d47b69fc73f0f2205be048'
cheked_year = '2015'
cheked_year_end = '2020'

quarter_limit = 1

years_len = (int(cheked_year_end)-int(cheked_year)) * quarter_limit





# tickers = si.tickers_sp500()
# print(tickers)

# Файл, полученный в Google Developer Console
CREDENTIALS_FILE = 'Seetzzz-1cb93f64d8d7.json'
# ID Google Sheets документа (можно взять из его URL)
spreadsheet_id = '1lDhu6-tBmoh66a1mY3RU2yPV2_3uIzNSQWNI5UtMcag'
spreadsheet_id2 = '1A3leW6ZfsoVEPXZsv0Loj4eAbyKRchnHrJLdP4RIXDA'
#
# Авторизуемся и получаем service — экземпляр доступа к API
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    CREDENTIALS_FILE,
    ['https://www.googleapis.com/auth/spreadsheets',
     'https://www.googleapis.com/auth/drive'])
httpAuth = credentials.authorize(httplib2.Http())
service = apiclient.discovery.build('sheets', 'v4', http=httpAuth)

# ____________________________Парсим тикеры !!!!С ТАБЛИЦЫ!!!! и работаем с ними _______________________________________

# Чтения файла
values = service.spreadsheets().values().get(
    spreadsheetId=spreadsheet_id,
    range=f'{LIST}!A1:L1000',
    majorDimension='COLUMNS'
).execute()

tickers = values['values'][1][1:]

print(tickers)
# tickers = tickers[:20]

price_yahoo = yf.download(tickers)
price_yahoo = price_yahoo['Adj Close'].fillna(method='backfill')

# tickers = ['AAPL', 'A']
is_main_df = pd.DataFrame()
km_main_df = pd.DataFrame()
bal_main_df = pd.DataFrame()
fr_main_df = pd.DataFrame()
ev_main_df = pd.DataFrame()
cf_main_df = pd.DataFrame()

for ticker in price_yahoo.columns.tolist():
    try:
        IS = requests.get(
            f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&apikey={KEY}').json()
        KM = requests.get(
            f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=quarter&apikey={KEY}').json()
        BL = requests.get(
            f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=quarter&apikey={KEY}').json()
        CF = requests.get(
            f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period=quarter&apikey={KEY}').json()
        EV = requests.get(
            f'https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}?period=quarter&apikey={KEY}').json()
        FR = requests.get(
            f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?period=quarter&apikey={KEY}').json()
        KM = requests.get(
            f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=quarter&apikey={KEY}').json()

        is_df = pd.DataFrame(IS).set_index('date')[cheked_year_end:cheked_year][::-1].set_index('symbol')
        sum_frame_key = [is_df, is_main_df]
        is_main_df = pd.concat(sum_frame_key)

        #         km_df = pd.DataFrame(KM).set_index('date')[cheked_year_end:cheked_year][::-1].set_index('symbol')
        km_df = pd.DataFrame(KM)
        km_df['Date'] = km_df['date']
        km_df = km_df.set_index('date')[cheked_year_end:cheked_year][::-1].set_index('symbol')

        sum_frame_km = [km_df, km_main_df]
        km_main_df = pd.concat(sum_frame_km)

        bal_df = pd.DataFrame(BL).set_index('date')[cheked_year_end:cheked_year][::-1].set_index('symbol')
        sum_frame_bal = [bal_df, bal_main_df]
        bal_main_df = pd.concat(sum_frame_bal)

        fr_df = pd.DataFrame(FR).set_index('date')[cheked_year_end:cheked_year][::-1].set_index('symbol')
        sum_frame_fr = [fr_df, fr_main_df]
        fr_main_df = pd.concat(sum_frame_fr)

        cf_df = pd.DataFrame(CF).set_index('date')[cheked_year_end:cheked_year][::-1].set_index('symbol')
        sum_frame_cf = [cf_df, cf_main_df]
        cf_main_df = pd.concat(sum_frame_cf)

        ev_df = pd.DataFrame(EV).set_index('date')[cheked_year_end:cheked_year][::-1].set_index('symbol')
        sum_frame_ev = [ev_df, ev_main_df]
        ev_main_df = pd.concat(sum_frame_ev)

    except:
        pass

# main_df.iloc['symbol']
# tickers[0]

# bal_main_df.loc[tickers[1]].fillna(0).iloc[2020 - int(cheked_year)+2]

portfolio_profit_final = []
index_profit_final = []
max_dd_list = []

quarter_len = int((int(cheked_year_end)-int(cheked_year))*4/quarter_limit)

all_tickers_check = []

ticker_check = ''
ticker_max_len = 0
for ticker in price_yahoo.columns.tolist():
    try:
        test = km_main_df.loc[ticker]
        all_tickers_check.append(ticker)
        if len(km_main_df.loc[ticker]['period']) > ticker_max_len:
            ticker_max_len = len(km_main_df.loc[ticker])
            ticker_check = ticker
    except:
        pass
print(ticker_check)
print(km_main_df.loc[ticker_check])

for i in range(len(km_main_df.loc[ticker_check]['period'])//quarter_limit):
    print('Itteration')
    print(i)

    # list_of_tickers = ['COG', 'AAPL', 'FB', 'INTC', 'MMM', 'A', 'BABA', 'XOM']
    list_of_tickers = tickers
    df_res = pd.DataFrame()

    for ticker in all_tickers_check:

        try:
            #             IS = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={KEY}').json()
            #             KM = requests.get(f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={KEY}').json()
            #             BL = requests.get(
            #                 f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?apikey={KEY}').json()
            #             CF = requests.get(
            #                 f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?apikey={KEY}').json()
            #             EV = requests.get(
            #                 f'https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}?&apikey={KEY}').json()
            #             FR = requests.get(
            #                 f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?&apikey={KEY}').json()
            #             KM = requests.get(f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={KEY}').json()

            #             km_main_df.loc['AAL'][quarter_limit*step:quarter_limit*(step+1)]

            key_metrics = km_main_df.loc[ticker].fillna(0)[quarter_limit * i:quarter_limit * (i + 1)].sum()
            balance = bal_main_df.loc[ticker].fillna(0)[quarter_limit * i:quarter_limit * (i + 1)].sum()
            financial_ratios = fr_main_df.loc[ticker].fillna(0)[quarter_limit * i:quarter_limit * (i + 1)].sum()
            income_statement = is_main_df.loc[ticker].fillna(0)[quarter_limit * i:quarter_limit * (i + 1)].sum()
            cash_flow = cf_main_df.loc[ticker].fillna(0)[quarter_limit * i:quarter_limit * (i + 1)].sum()
            enterprise_value = ev_main_df.loc[ticker].fillna(0)[quarter_limit * i:quarter_limit * (i + 1)].sum()

            ### Value Factor ###
            Data_for_Portfolio = pd.DataFrame()

            Data_for_Portfolio['E/P'] = [income_statement['netIncome'] / enterprise_value['marketCapitalization']]
            Data_for_Portfolio['EBITDA/EV'] = income_statement['ebitda'] / \
                                              enterprise_value['enterpriseValue']
            Data_for_Portfolio['FCF/P'] = cash_flow['freeCashFlow'] / \
                                          enterprise_value['marketCapitalization']

            Data_for_Portfolio['ncfcommon'] = cash_flow['freeCashFlow'] / enterprise_value['numberOfShares']

            ### Shareholder Yield ###
            Data_for_Portfolio['Shareholder Yield'] = \
                -((financial_ratios['cashFlowToDebtRatio'] + \
                   cash_flow['freeCashFlow'] + \
                   Data_for_Portfolio['ncfcommon']) / enterprise_value['marketCapitalization'])

            ### Quality Factor - ideas taken from Alpha Architect QV model ####
            ####Long Term Business Strength
            # Can you generate free cash flow?
            Data_for_Portfolio['FCF/Assets'] = cash_flow['freeCashFlow'] / \
                                               balance['totalCurrentAssets']
            # Can you generate returns on investment?
            Data_for_Portfolio['ROA'] = income_statement['netIncome'] / balance['totalAssets']
            Data_for_Portfolio['ROIC'] = key_metrics['roic']
            # Do you have a defendable business model?
            Data_for_Portfolio['GROSS MARGIN'] = income_statement['grossProfit'] / income_statement['revenue']
            # Current Financial Strength
            Data_for_Portfolio['CURRENT RATIO'] = key_metrics['currentRatio']
            Data_for_Portfolio['INTEREST/EBITDA'] = income_statement['interestExpense'] / income_statement['ebitda']
            Data_for_Portfolio['Company'] = ticker

            sum_frame = [Data_for_Portfolio, df_res]
            df_res = pd.concat(sum_frame)
        #         print('Norm')
        except:
            #         print('NE Norm')
            pass

    Data_for_Portfolio = df_res.set_index('Company')
    Data_for_Portfolio_master_filter = Data_for_Portfolio = df_res.set_index('Company').replace([np.inf, -np.inf], 0)

    t0 = time.time()

    Data_for_Portfolio = Data_for_Portfolio.dropna()

    print(Data_for_Portfolio_master_filter)


    # Using the same in sample dates here and for equal weight benchmark

    #     f_date = datetime.date(2000, 9, 30)
    #     l_date = datetime.date(2012, 9, 30) #choosing the last date, results in last
    # date for returns is l_date + 1 quarter

    #     delta = l_date - f_date
    #     quarters_delta = np.floor(delta.days/(365/4))
    #     quarters_delta = int(quarters_delta)
    #     first_quarter = str(int(cheked_year)+i) #using f_date

    Data_for_Portfolio_master = pd.DataFrame(Data_for_Portfolio)

    # choose if you want percentiles or fixed number of companies in long portfolio
    Percentile_split = .2

    Winsorize_Threshold = .025  # used to determine the winsorize level. If you are
    # only going to have a handful of companies than put the threshold really low,
    # otherwise you can use around .025 for a decile portfolio
    Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.replace([np.inf, -np.inf], 0)
    ###### VALUE FACTOR ######

    Data_for_Portfolio_master_filter = Data_for_Portfolio_master
    # Winsorize the metric data and compress outliers if desired
    Data_for_Portfolio_master_filter['E/P Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['E/P'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['EBITDA/EV Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['EBITDA/EV'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['FCF/P Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['FCF/P'], \
                               limits=Winsorize_Threshold)

    # create Z score to normalize the metrics
    Data_for_Portfolio_master_filter['E/P Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['E/P Winsorized'])
    Data_for_Portfolio_master_filter['EBITDA/EV Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['EBITDA/EV Winsorized'])
    Data_for_Portfolio_master_filter['FCF/P Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['FCF/P Winsorized'])

    Data_for_Portfolio_master_filter['Valuation Score'] = \
        Data_for_Portfolio_master_filter['E/P Z score'] \
        + Data_for_Portfolio_master_filter['EBITDA/EV Z score'] \
        + Data_for_Portfolio_master_filter['FCF/P Z score']

    Data_for_Portfolio_master_filter['FCF/P Winsorized']

    ###### QUALITY FACTOR ######
    Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.replace([np.inf, -np.inf], 0)

    Data_for_Portfolio_master_filter['FCF/Assets Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['FCF/Assets'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['ROA Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['ROA'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['ROIC Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['ROIC'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['Gross Margin Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['GROSS MARGIN'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['Current Ratio Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['CURRENT RATIO'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['Interest/EBITDA Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['INTEREST/EBITDA'], \
                               limits=Winsorize_Threshold)

    # create Z score
    Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.replace([np.inf, -np.inf], 0)

    Data_for_Portfolio_master_filter['FCF/Assets Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['FCF/Assets Winsorized'])
    Data_for_Portfolio_master_filter['ROA Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['ROA Winsorized'])
    Data_for_Portfolio_master_filter['ROIC Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['ROIC Winsorized'])
    Data_for_Portfolio_master_filter['Gross Margin Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['Gross Margin Winsorized'])
    Data_for_Portfolio_master_filter['Current Ratio Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['Current Ratio Winsorized'])
    Data_for_Portfolio_master_filter['Interest/EBITDA Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['Interest/EBITDA Winsorized'])

    Data_for_Portfolio_master_filter['Quality Score'] = \
        Data_for_Portfolio_master_filter['FCF/Assets Z score'] \
        + Data_for_Portfolio_master_filter['ROA Z score'] \
        + Data_for_Portfolio_master_filter['ROIC Z score'] \
        + Data_for_Portfolio_master_filter['Gross Margin Z score'] \
        + Data_for_Portfolio_master_filter['Current Ratio Z score'] \
        - Data_for_Portfolio_master_filter['Interest/EBITDA Z score']

    ###### SHAREHOLDER YIELD FACTOR #####

    Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.replace([np.inf, -np.inf], 0)

    Data_for_Portfolio_master_filter['Shareholder Yield Winsorized'] = \
        stats.mstats.winsorize(Data_for_Portfolio_master_filter['Shareholder Yield'], \
                               limits=Winsorize_Threshold)
    Data_for_Portfolio_master_filter['Shareholder Yield Z score'] = \
        stats.zscore(Data_for_Portfolio_master_filter['Shareholder Yield Winsorized'])
    Data_for_Portfolio_master_filter['Shareholder Yield Score'] = \
        Data_for_Portfolio_master_filter['Shareholder Yield Z score']

    ###### LOW VOLATILITY FACTOR ######

    try:
        # must have fundamental data from previous factors for price based factors
        # as some equities have price data and no fundamental data which should not
        # be included

        # treasury = 'RGBI.ME'
        start = cheked_year
        # end = current_date
        end = cheked_year_end

        price_yahoo = yf.download(Data_for_Portfolio_master_filter.index.tolist())
        price_yahoo = price_yahoo['Adj Close'].fillna(method='backfill')

        Sector_stock_returns = price_yahoo.pct_change()

        # create rolling vol metric for previous 2 years
        Sector_stock_rolling_vol = Sector_stock_returns.rolling(252 * 2).std()

        # Choose second to last trading day to look at previous vol
        # Sometimes the dates are off when trying to line up end of quarter and business
        # days so to eliminate errors in the for loop I go to day of quarter, shift forward
        # a business day and then go back two business days

        #     Filter_Vol_Signal = Sector_stock_rolling_vol[str(int(start)+i)]
        print(km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)])
        print(km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)])

        Filter_Vol_Signal = Sector_stock_rolling_vol[km_main_df.loc[ticker_check] \
                                                         [quarter_limit * i:quarter_limit * (i + 1)]['Date'].values[0]: \
                                                     km_main_df.loc[ticker_check][
                                                     quarter_limit * i:quarter_limit * (i + 1)]['Date'].values[-1]]

        # Filter_Vol_Signal_Sort = Filter_Vol_Signal.sort_values().dropna() # для цикла паска тикеров

        # create z score and rank for the Volatility Factor
        # frame = { 'Vol': Filter_Vol_Signal}

        # Filter_Vol_Signal_df = pd.DataFrame(frame)

        #     print(pd.DataFrame(stats.zscore(Filter_Vol_Signal)).mean())
        #     print(pd.DataFrame(stats.zscore(Filter_Vol_Signal)))

        # Filter_Vol_Signal_df['Vol Z Score'] = stats.zscore(Filter_Vol_Signal)
        # Filter_Vol_Signal_df = Filter_Vol_Signal_df.reset_index()
        # print(Filter_Vol_Signal_df)

        Data_for_Portfolio_master_filter['Vol Z Score'] = pd.DataFrame(stats.zscore(Filter_Vol_Signal)).mean().tolist()
        # Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.merge(Filter_Vol_Signal_df, how = 'inner', on = ['ticker'])
        Data_for_Portfolio_master_filter

    except:
        Data_for_Portfolio_master_filter['Vol Z Score'] = [0]

    try:
        ###### TREND FACTOR #####

        # This is a very simply way to see how much a stock is in a trend up or down
        # You could easily make this more complex/robust but it would cost you in
        # execution time
        df_sma_50 = price_yahoo.rolling(50).mean()
        df_sma_100 = price_yahoo.rolling(100).mean()
        df_sma_150 = price_yahoo.rolling(150).mean()
        df_sma_200 = price_yahoo.rolling(200).mean()

        # print(df_sma_200)

        # Get the same date for vol measurement near rebalance date

        # Filter_Date_Trend = final_trade_date.strftime('%Y-%m-%d')

        Filter_Trend_Signal_50 = df_sma_50[km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                               ['Date'].values[0]:
                                           km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                               ['Date'].values[-1]]
        Filter_Trend_Signal_100 = df_sma_100[km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                                 ['Date'].values[0]:
                                             km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                                 ['Date'].values[-1]]
        Filter_Trend_Signal_150 = df_sma_150[km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                                 ['Date'].values[0]:
                                             km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                                 ['Date'].values[-1]]
        Filter_Trend_Signal_200 = df_sma_200[km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                                 ['Date'].values[0]:
                                             km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                                 ['Date'].values[-1]]

        Price_Signal = price_yahoo[km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                       ['Date'].values[0]:
                                   km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)] \
                                       ['Date'].values[-1]]

        SMA_all = pd.DataFrame()
        # Filter_SMA_Signal_df = Filter_SMA_Signal_df.rename(columns={0: "ticker"})
        # print(Filter_SMA_Signal_df)
        SMA_50 = pd.DataFrame(np.where(Price_Signal > Filter_Trend_Signal_50, 1, 0)).mean()
        # print(np.where(Price_Signal > Filter_Trend_Signal_50,1,0))
        SMA_100 = pd.DataFrame(np.where(Price_Signal > Filter_Trend_Signal_100, 1, 0)).mean()
        SMA_150 = pd.DataFrame(np.where(Price_Signal > Filter_Trend_Signal_150, 1, 0)).mean()
        SMA_200 = pd.DataFrame(np.where(Price_Signal > Filter_Trend_Signal_200, 1, 0)).mean()

        SMA_all['SMA_50'] = SMA_50
        SMA_all['SMA_100'] = SMA_100
        SMA_all['SMA_150'] = SMA_150
        SMA_all['SMA_200'] = SMA_200
        SMA_all['Trend Score'] = np.mean(SMA_all, axis=1)

        # print(SMA_all)
        Data_for_Portfolio_master_filter['Trend Score'] = np.mean(SMA_all, axis=1).tolist()

    except:
        Data_for_Portfolio_master_filter['Trend Score'] = [0]

    try:
        ###### MOMENTUM FACTOR #####

        #     print('tut')

        # tickers_momentum = list(Sector_stock_prices_vol_df_1_wide.columns)
        # from the academic literature of 12 months - 1 month momentum
        df_mom_11_months = price_yahoo[
                           km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)]['Date'].values[0]:
                           km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)]['Date'].values[-1]]
        df_mom_11_months = df_mom_11_months.pct_change(len(df_mom_11_months) - 22)
        # print(df_mom_11_months.head())
        # Filter_Date_Mom = Date_to_execute_trade_plus1 - pd.tseries.offsets.BusinessDay(24)
        # Filter_Date_Mom_trim = final_trade_date.strftime('%Y-%m-%d')
        # Filter_Mom_Signal = df_mom_11_months.loc[Filter_Date_Mom_trim]

        # print(np.mean(stats.zscore(df_mom_11_months.dropna()), axis=0))
        #     print('tut2')

        # Filter_MOM_df = pd.DataFrame(ticker)
        # Filter_MOM_df = Filter_MOM_df.rename(columns={0: "ticker"})
        # Filter_MOM_df['Percent Change'] = Filter_Mom_Signal.values

        # Filter_MOM_df = Filter_MOM_df.replace([np.inf, -np.inf], np.nan)
        # Filter_MOM_df = Filter_MOM_df.dropna()
        # Filter_MOM_df['Momentum Score'] = stats.zscore(Filter_MOM_df['Percent Change'])

        Data_for_Portfolio_master_filter['Momentum Score'] = np.mean(
            stats.zscore(df_mom_11_months.iloc[len(df_mom_11_months) - 22:].fillna(0)), axis=0)
        # Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.merge(Filter_MOM_df[['ticker','Momentum Score']], how = 'inner', on = ['ticker'])

    except:
        Data_for_Portfolio_master_filter['Momentum Score'] = [0]

    ### Create Composite Score from factors ###

    # Because we made all the factors with a z score each factor should have equal
    # weight in the composite. You could consider changing the weights based on
    # historical statistical significance or whatever else seems reasonable

    # This particular scoring system only invests in companies with
    # positive trend/momentum after ranking by the other factors

    Data_for_Portfolio_master_filter['Total Score'] = \
        Data_for_Portfolio_master_filter['Valuation Score'] + \
        Data_for_Portfolio_master_filter['Quality Score'] + \
        Data_for_Portfolio_master_filter['Shareholder Yield Score'] - \
        Data_for_Portfolio_master_filter['Vol Z Score'] * \
        (Data_for_Portfolio_master_filter['Momentum Score'] + \
         Data_for_Portfolio_master_filter['Trend Score'])

    #     start_hayoo = str(int(start)+i+1)+'-1-1'
    #     end_hayoo = str(int(end)+i+1)+'-1-1'

    try:
        cum_str_returns_bh = (price_yahoo[
                              km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)]['Date'].values[
                                  -1]:km_main_df.loc[ticker_check][quarter_limit * (i + 1):quarter_limit * (i + 2)][
                                  'Date'].values[-1]].fillna(method='backfill').pct_change() + 1).cumprod()
        running_max_BH = np.maximum.accumulate(cum_str_returns_bh[1:].fillna(method='backfill'))
        drawdown_BH = (cum_str_returns_bh[1:]) / running_max_BH - 1
        max_dd = drawdown_BH.min() * 100

        Data_for_Portfolio_master_filter['Max DD'] = max_dd.values

        max_dd_list.append(max_dd.min())

    except:
        pass

    # number_firms = Data_for_Portfolio_master_filter.shape
    # number_firms = number_firms[0]

    # firms_in_percentile = np.round(Percentile_split * number_firms)

    Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.sort_values('Total Score', ascending=False)

    top_rated_company = Data_for_Portfolio_master_filter[:int(len(Data_for_Portfolio_master_filter) \
                                                              * Percentile_split)].index.tolist()
    top_rated_company

    low_rated_company = Data_for_Portfolio_master_filter[-int(len(Data_for_Portfolio_master_filter) \
                                                              * Percentile_split):].index.tolist()
    low_rated_company

    #     print('tut3')

    #     # == Доходность

    #     start_hayoo = str(int(start)+i+1)+'-1-1'
    #     end_hayoo = str(int(end)+i+1)+'-1-1'

    print(start)

    portfolio_profit = []
    profit_list_index = 0

    for company in top_rated_company:
        #         print('tut1')

        try:
            profit_yah = yf.download(company, km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)][
                'Date'].values[-1], km_main_df.loc[ticker_check][quarter_limit * (i + 1):quarter_limit * (i + 2)][
                                         'Date'].values[-1])['Adj Close'].fillna(method='backfill')
            profit = (profit_yah[-1] - profit_yah[0]) / profit_yah[0]
            (1 + profit).cumprod()[-1]
            portfolio_profit.append(profit)
            print(profit)
        except:
            pass

    try:
        portfolio_profit

        profit_list_index = 0

        profit_yah = \
        yf.download(index, km_main_df.loc[ticker_check][quarter_limit * i:quarter_limit * (i + 1)]['Date'].values[-1],
                    km_main_df.loc[ticker_check][quarter_limit * (i + 1):quarter_limit * (i + 2)]['Date'].values[-1])[
            'Adj Close'].fillna(method='backfill')
        profit = (profit_yah[-1] - profit_yah[0]) / profit_yah[0]

        #         print(start_hayoo)

        profit_list_index = profit

        portfolio_profit_final.append(np.mean(portfolio_profit) * 100)
        index_profit_final.append(profit_list_index * 100)

        returnezzz = pd.DataFrame()
        returnezzz['Portfolio'] = [np.mean(portfolio_profit) * 100]
        returnezzz['Index'] = [profit_list_index * 100]

    except:
        pass
    print('portfolio_profit_final')
    print(portfolio_profit_final)

    print('top_rated_company')
    print(top_rated_company)
    print('low_rated_company')
    print(low_rated_company)

    print('Max DD')
    print(max_dd_list)
    print(np.min(max_dd_list))


returnez_cum_port = pd.DataFrame(portfolio_profit_final)
returnez_cum_index = pd.DataFrame(index_profit_final)

returnez = pd.DataFrame()

returnez['Страна'] = [LIST]
returnez['Начало периода'] = [cheked_year]
returnez['Дходность с ребалансировкой портфеля'] = ((1 + (returnez_cum_port/100)).cumprod().iloc[-1]-1)*100
returnez['Дходность Индекса'] = ((1 + (returnez_cum_index/100)).cumprod().iloc[-1]-1)*100
returnez['Max DD'] = [np.min(max_dd_list)]


# gc = gd.service_account(filename='Seetzzz-1cb93f64d8d7.json')
# worksheet = gc.open("Тесты бэктестинга").worksheet('Мульти-фактор2')

# worksheet.update('A20', [returnez.columns.tolist()] + returnez.values.tolist())

returnez.head()
# print()

gc = gd.service_account(filename='Seetzzz-1cb93f64d8d7.json')
worksheet = gc.open("Тесты бэктестинга").worksheet('Мульти-фактор2')

worksheet.update('H2', [returnez.columns.tolist()] + returnez.values.tolist())