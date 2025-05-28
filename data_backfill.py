# /opt/anaconda3/bin/activate && conda activate /opt/anaconda3/envs/KiteConnect

# 0 16 * * * cd /Users/shashankbaveja/Downloads/Personal/KiteConnectAPI/trading_setup && /opt/anaconda3/envs/KiteConnect/bin/python data_backfill.py >> /Users/shashankbaveja/Downloads/Personal/KiteConnectAPI/trading_setup/data_backfill_cron.log 2>&1

from IPython import embed;
from kiteconnect import KiteConnect, KiteTicker
import mysql
import mysql.connector as sqlConnector
import datetime
from selenium import webdriver
import os
from pyotp import TOTP
import ast
import time
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from myKiteLib import system_initialization
from myKiteLib import kiteAPIs
import logging
import json
from datetime import date, timedelta, datetime, time
from kiteconnect.exceptions import KiteException  # Import KiteConnect exceptions
import requests # Import requests for ReadTimeout

from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


# if __name__ == "__main__":
#     BACKFILL_INTERVAL = 'minute'
#     BACKFILL_DAYS = 10
    
#     today_date = date.today()
    
#     end_dt_backfill = datetime.combine(today_date, time(23, 59, 59))
    
#     start_date_val = today_date - timedelta(days=BACKFILL_DAYS)
#     start_dt_backfill = datetime.combine(start_date_val, time(0, 0, 0))

#     systemDetails = system_initialization()
#     systemDetails.init_trading()
#     callKite = kiteAPIs()

#     tokenList = [256265] ## NIFTY INDEX
#     # tokenList.extend(callKite.get_instrument_active_tokens('CE',end_dt_backfill))
#     # tokenList.extend(callKite.get_instrument_active_tokens('PE',end_dt_backfill))
#     # tokenList.extend(callKite.get_instrument_active_tokens('FUT',end_dt_backfill)
#     tokenList.extend(callKite.get_instrument_all_tokens('EQ'))


#     try:
#         df = callKite.getHistoricalData(start_dt_backfill,  end_dt_backfill, tokenList, BACKFILL_INTERVAL)
#     except (KiteException, requests.exceptions.ReadTimeout) as e:
#         print(f"Error fetching historical data: {e}")
#         logging.error(f"Error fetching historical data: {e}")
#         df = pd.DataFrame() # Initialize an empty DataFrame or handle as needed



    


if __name__ == "__main__":
    BACKFILL_INTERVAL = 'minute'
    BACKFILL_DAYS = 59
    
    # today_date = date.today()
    today_date = date(2024,10,4)
    end_dt_backfill = datetime.combine(today_date, time(23, 59, 59))
    
    start_date_val = today_date - (timedelta(days=BACKFILL_DAYS))
    start_dt_backfill = datetime.combine(start_date_val, time(0, 0, 0))
    
    for i in range(1,30):
        print(f"Fetching data for {start_dt_backfill} to {end_dt_backfill}")
        systemDetails = system_initialization()
        systemDetails.init_trading()
        callKite = kiteAPIs()

        tokenList = [256265] ## NIFTY INDEX
        # tokenList.extend(callKite.get_instrument_active_tokens('CE',end_dt_backfill))
        # tokenList.extend(callKite.get_instrument_active_tokens('PE',end_dt_backfill))
        # tokenList.extend(callKite.get_instrument_active_tokens('FUT',end_dt_backfill)
        tokenList.extend(callKite.get_instrument_all_tokens('EQ'))


        try:
            df = callKite.getHistoricalData(start_dt_backfill,  end_dt_backfill, tokenList, BACKFILL_INTERVAL)
        except (KiteException, requests.exceptions.ReadTimeout) as e:
            print(f"Error fetching historical data: {e}")
            logging.error(f"Error fetching historical data: {e}")
            df = pd.DataFrame() # Initialize an empty DataFrame or handle as needed
        end_dt_backfill = end_dt_backfill - (timedelta(days=BACKFILL_DAYS))
        start_dt_backfill = start_dt_backfill - (timedelta(days=BACKFILL_DAYS))



    
