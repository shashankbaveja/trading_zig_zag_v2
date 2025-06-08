
from IPython import embed;
from kiteconnect import KiteConnect, KiteTicker
import mysql
import mysql.connector as sqlConnector
import datetime
from selenium import webdriver
import os
from pyotp import TOTP
import ast
import time as t
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




BACKFILL_INTERVAL = 'minute'

today_date = date.today()

end_dt_backfill = datetime.combine(today_date, time(23, 59, 59))

start_date_val = today_date
start_dt_backfill = datetime.combine(start_date_val, time(0, 0, 0))

systemDetails = system_initialization()
systemDetails.init_trading()
callKite = kiteAPIs()

tokenList = [256265] ## NIFTY INDEX
i = 0
while i < 1000:
    try:
        df = callKite.getHistoricalData(start_dt_backfill,  end_dt_backfill, tokenList, BACKFILL_INTERVAL)
    except (KiteException, requests.exceptions.ReadTimeout) as e:
        print(f"Error fetching historical data: {e}")
        logging.error(f"Error fetching historical data: {e}")
        df = pd.DataFrame() # Initialize an empty DataFrame or handle as needed

    df = df[['date','open','high','low','close','volume']]
    print(df.tail(1))
    i = i + 1
    t.sleep(10)
