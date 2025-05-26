import json
from kiteconnect import KiteConnect, KiteTicker
import mysql
import numpy as np
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
import requests
from IPython import embed
from kiteconnect.exceptions import KiteException  # Import KiteConnect exceptions
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt

telegramToken_global = '8135376207:AAFoMWbyucyPPEzc7CYeAMTsNZfqHWYDMfw' # Renamed to avoid conflict
telegramChatId_global = "-4653665640"

class system_initialization:
    def __init__(self):
        
        self.Kite = None
        self.con = None

        # Telegram credentials for instance use
        self.telegram_token = telegramToken_global
        self.telegram_chat_id = telegramChatId_global

        config_file_path = "./security.txt"
        with open(config_file_path, 'r') as file:
            content = file.read()

        self.config = ast.literal_eval(content)
        self.api_key = self.config["api_key"]
        self.api_secret = self.config["api_secret"]
        self.userId = self.config["userID"]
        self.pwd = self.config["pwd"]
        self.totp_key = self.config["totp_key"]
        
        self.mysql_username = self.config["username"]
        self.mysql_password = self.config["password"]
        self.mysql_hostname = self.config["hostname"]
        self.mysql_port = int(self.config["port"])
        self.mysql_database_name = self.config["database_name"]
        self.AccessToken = self.config["AccessToken"]

        print("read security details")

        self.kite = KiteConnect(api_key=self.api_key, timeout=60)
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
    
    
    def init_trading(self):
        self.kite.set_access_token(self.AccessToken)
        try:
            data = self.kite.historical_data(256265,'2025-05-15','2025-05-15','minute')
        except KiteException as e:
            print(e)
            print("Access token expired, Generating new token")
            temp_token = self.kite_chrome_login_generate_temp_token()
            AccessToken = self.kite.generate_session(temp_token, api_secret= self.api_secret)["access_token"]
            self.kite.set_access_token(AccessToken)
            self.SaveAccessToken(AccessToken)

            # Update the in-memory configuration and write to security.txt
            self.config["AccessToken"] = AccessToken # Update self.config dict
            config_file_path = "./security.txt"
            try:
                with open(config_file_path, 'w') as file:
                    json.dump(self.config, file, indent=4) # Write the whole updated config
                print("Successfully updated AccessToken in security.txt")
            except Exception as update_err:
                print(f"Error updating AccessToken in security.txt: {update_err}")
            
            AccessToken = self.GetAccessToken()
            self.kite.set_access_token(AccessToken)        

        df_nse = self.download_instruments('NSE')
        df_nfo = self.download_instruments('NFO')
        df=pd.concat([df_nse, df_nfo], axis=0)
        print("starting DB save")
        self.save_intruments_to_db(data = df)

        print('Ready to trade')
        # Send initial ready message using the new instance method if OrderPlacement has it,
        # or keep it here if system_initialization itself should send it.
        # For now, let's assume a specific ready message might be sent by LiveTrader itself.
        # Original telegram message send:
        # telegramMessage = 'Ready to trade'
        # telegramURL = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(self.telegram_token,self.telegram_chat_id,telegramMessage)
        # response = requests.get(telegramURL)
        return self.kite

    def GetAccessToken(self):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        cursor = self.con.cursor()
        query = "Select token from daily_token_log where date(created_at) = '{}' order by created_at desc limit 1;".format(datetime.date.today())
        cursor.execute(query)
        print("read token from DB")
        for row in cursor:
            if row is None:
                return ''
            else:
                return row[0]
        self.con.close()
    def run_query_limit_1(self,query):
        cursor = self.con.cursor()
        cursor.execute(query)
        print("read token from DB")
        for row in cursor:
            if row is None:
                return ''
            else:
                return row[0]
    def SaveAccessToken(self,Token):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        q1 = "INSERT INTO kiteConnect.daily_token_log(token) Values('{}')"
        q2 = " ON DUPLICATE KEY UPDATE Token = '{}', created_at=CURRENT_TIMESTAMP();"
        q1 = q1.format(Token)
        q2 = q2.format(Token)
        query = q1 + q2
        cur = self.con.cursor()
        cur.execute(query)
        self.con.commit()
        print("saved token to DB")
        self.con.close()


    def kite_chrome_login_generate_temp_token(self):
        browser = webdriver.Chrome()
        browser.get(self.kite.login_url())
        browser.implicitly_wait(5)

        username = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[1]/input')
        password = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[2]/input') 
        
        username.send_keys(self.userId)
        password.send_keys(self.pwd)
        
        # Click Login Button
        browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[4]/button').click()
        time.sleep(2)

        pin = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div[2]/div/div[2]/form/div[1]/input')
        totp = TOTP(self.totp_key)
        token = totp.now()
        pin.send_keys(token)
        time.sleep(1)
        temp_token=browser.current_url.split('request_token=')[1][:32]
        browser.close()

        print("got temp token")
        
        return temp_token

    def download_instruments(self, exch):
        lst = []
        if exch == 'NSE':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_NSE)
        elif exch == 'NFO':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_NFO) # derivatives NSE
        elif exch == 'BSE':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_BSE)
        elif exch =='CDS':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_CDS) # Currency
        elif exch == 'BFO':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_BFO) # Derivatives BSE
        elif exch == 'MCX':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_MCX) # Commodity
        else:
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_BCD)
            
        df = pd.DataFrame(lst) # Convert list to dataframe
        if len(df) == 0:
            print('No data returned')
            return
        print("downloading instruments")
        return df

    def save_data_to_db(self, data, tableName):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        print("starting DB save - entered function")
        engine = create_engine("mysql+pymysql://{user}:{pw}@{localhost}:{port}/{db}".format(user=self.mysql_username, localhost = self.mysql_hostname, port = self.mysql_port, pw=self.mysql_password, db=self.mysql_database_name))
        print("starting DB save - created engine")
        data.to_sql(tableName, con = engine, if_exists = 'replace', chunksize = 100000)
        print('Saved to Database')
        self.con.close()

    def save_intruments_to_db(self,data):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        for i in range (0, len(data)):
            query = "insert into kiteConnect.instruments_zerodha values({},'{}','{}','{}',{},'{}',{},{},{},'{}','{}','{}') ON DUPLICATE KEY UPDATE instrument_token=instrument_token;".format(data['instrument_token'].iloc[i],data['exchange_token'].iloc[i],data['tradingsymbol'].iloc[i],data['name'].iloc[i],data['last_price'].iloc[i],data['expiry'].iloc[i],data['strike'].iloc[i],data['tick_size'].iloc[i],data['lot_size'].iloc[i],data['instrument_type'].iloc[i],data['segment'].iloc[i],data['exchange'].iloc[i])
            cur = self.con.cursor()
            cur.execute(query)
            self.con.commit()
            # print("saved token to DB")
        self.con.close()

class OrderPlacement(system_initialization):
    def __init__(self):
        super().__init__() # Initialize the base class to get self.kite, etc.
        self.k_apis = kiteAPIs()  # Create instance of kiteAPIs
        print("OrderPlacement module initialized. Ensure init_trading() is called if access token needs refresh.")

    def send_telegram_message(self, message: str):
        """
        Sends a message to the configured Telegram chat.
        Args:
            message (str): The message text to send.
        """
        if not self.telegram_token or not self.telegram_chat_id:
            print("OrderPlacement: Telegram token or chat ID not configured. Cannot send message.")
            return
        
        telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        params = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'Markdown' # Optional: for formatting like *bold* or _italic_
        }
        try:
            response = requests.get(telegram_url, params=params, timeout=10) # Added timeout
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
            print(f"OrderPlacement: Telegram message sent successfully: '{message[:50]}...'")
        except requests.exceptions.RequestException as e:
            print(f"OrderPlacement: Error sending Telegram message: {e}")
        except Exception as e:
            print(f"OrderPlacement: A general error occurred sending Telegram message: {e}")

    def place_market_order_live(self, tradingsymbol: str, exchange: str, transaction_type: str, 
                                quantity: int, product: str, tag: str = None):
        """
        Places a market order.
        Args:
            tradingsymbol (str): Trading symbol of the instrument.
            exchange (str): Name of the exchange (e.g., self.kite.EXCHANGE_NFO, self.kite.EXCHANGE_NSE).
            transaction_type (str): Transaction type (self.kite.TRANSACTION_TYPE_BUY or self.kite.TRANSACTION_TYPE_SELL).
            quantity (int): Quantity to transact.
            product (str): Product code (e.g., self.kite.PRODUCT_MIS, self.kite.PRODUCT_NRML).
            tag (str, optional): An optional tag for the order.
        Returns:
            str: Order ID if successful, None otherwise.
        """
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=self.kite.ORDER_TYPE_MARKET,
                tag=tag
            )
            print(f"OrderPlacement: Market order placed for {tradingsymbol}. Order ID: {order_id}")
            return order_id
        except KiteException as e:
            print(f"OrderPlacement: Error placing market order for {tradingsymbol}: {e}")
            # Consider specific error handling or re-raising
        except Exception as e:
            print(f"OrderPlacement: A general error occurred placing market order for {tradingsymbol}: {e}")
        return None

    def get_order_history_live(self, order_id: str):
        """
        Retrieves the history of an order.
        Args:
            order_id (str): The ID of the order.
        Returns:
            list: A list of order updates (dicts) if successful, None otherwise.
        """
        try:
            history = self.kite.order_history(order_id=order_id)
            # print(f"OrderPlacement: Fetched history for order ID {order_id}.")
            return history
        except KiteException as e:
            print(f"OrderPlacement: Error fetching history for order ID {order_id}: {e}")
        except Exception as e:
            print(f"OrderPlacement: A general error occurred fetching history for order ID {order_id}: {e}")
        return None

    def get_all_orders_live(self):
        """
        Retrieves the list of all orders for the day.
        Returns:
            list: A list of order dicts if successful, None otherwise.
        """
        try:
            orders = self.kite.orders()
            # print(f"OrderPlacement: Fetched all orders. Count: {len(orders)}")
            return orders
        except KiteException as e:
            print(f"OrderPlacement: Error fetching all orders: {e}")
        except Exception as e:
            print(f"OrderPlacement: A general error occurred fetching all orders: {e}")
        return None

    def get_positions_live(self):
        """
        Retrieves the current open positions.
        Returns:
            dict: A dictionary with 'net' and 'day' positions if successful, None otherwise.
        """
        try:
            positions = self.kite.positions()
            # print(f"OrderPlacement: Fetched positions.")
            return positions
        except KiteException as e:
            print(f"OrderPlacement: Error fetching positions: {e}")
        except Exception as e:
            print(f"OrderPlacement: A general error occurred fetching positions: {e}")
        return None

    def get_trades_for_order_live(self, order_id: str):
        """
        Retrieves trades generated for a specific order.
        Args:
            order_id (str): The ID of the order.
        Returns:
            list: A list of trade dicts if successful, None otherwise.
        """
        try:
            trades = self.kite.order_trades(order_id=order_id)
            # print(f"OrderPlacement: Fetched trades for order ID {order_id}. Count: {len(trades)}")
            return trades
        except KiteException as e:
            print(f"OrderPlacement: Error fetching trades for order ID {order_id}: {e}")
        except Exception as e:
            print(f"OrderPlacement: A general error occurred fetching trades for order ID {order_id}: {e}")
        return None
    
    def get_ltp_live(self, instrument_token: int) -> float | None:
        """
        Fetches the Last Traded Price (LTP) for a given instrument token.
        Uses the exchange prefix NFO: as these are typically NFO option tokens.
        Args:
            instrument_token (int): The instrument token.
        Returns:
            float: The LTP if successful, None otherwise.
        """
        # KiteConnect's ltp method can take a list of instrument tokens directly (as int)
        # or strings like "EXCHANGE:TRADINGSYMBOL".
        # For NFO options, the integer token is usually sufficient.
        # The response is a dictionary where keys are "exchange:tradingsymbol" or the token string,
        # and values are dicts containing 'last_price'.
        
        # We need to construct the key that Kite API will return in the LTP dictionary.
        # This can be tricky if we only have the token.
        # A common format is 'NFO:TOKEN_AS_STRING' or just the token itself if it's an int list.
        # Let's try with the integer token first, as it's simpler.
        # The API expects a list of instrument tokens or instrument names.
        # Example: kite.ltp([123456, 789012]) or kite.ltp(["NFO:NIFTY23JUL18000CE", "NSE:RELIANCE"])

        try:
            # Ensure the kite session is initialized (it should be by OrderPlacement's __init__)
            if not self.kite:
                self.logger.error("OrderPlacement: Kite object not initialized for LTP fetch.")
                return None

            ltp_data = self.kite.ltp([instrument_token]) # Pass as a list
            # self.logger.debug(f"LTP raw response for {instrument_token}: {ltp_data}")

            if ltp_data and str(instrument_token) in ltp_data: # LTP keys are strings of tokens
                # The key in the response dictionary is the string representation of the instrument token
                # when a list of integer tokens is passed.
                token_data = ltp_data[str(instrument_token)]
                if token_data and 'last_price' in token_data:
                    ltp_value = float(token_data['last_price'])
                    # self.logger.debug(f"OrderPlacement: LTP for token {instrument_token}: {ltp_value}")
                    return ltp_value
                else:
                    self.logger.warning(f"OrderPlacement: 'last_price' not in LTP response for token {instrument_token}. Data: {token_data}")
            elif ltp_data:
                # Fallback: If the key is not the direct string of the token,
                # it might be because the API sometimes returns keys with exchange prefix.
                # This part is more complex as we don't have the exact tradingsymbol here.
                # For now, we rely on the direct token key.
                # Example: Check if any key in ltp_data corresponds to our token if it has an exchange prefix
                for key, value_dict in ltp_data.items():
                    if isinstance(value_dict, dict) and value_dict.get('instrument_token') == instrument_token:
                        if 'last_price' in value_dict:
                            ltp_value = float(value_dict['last_price'])
                            # self.logger.debug(f"OrderPlacement: LTP for token {instrument_token} (found via key {key}): {ltp_value}")
                            return ltp_value
                        else:
                            self.logger.warning(f"OrderPlacement: 'last_price' not in LTP response for token {instrument_token} (key {key}). Data: {value_dict}")
                            return None # Found the token but no last_price

                self.logger.warning(f"OrderPlacement: LTP data received, but token {instrument_token} not found directly or via iteration. LTP Data: {ltp_data}")

            else:
                self.logger.warning(f"OrderPlacement: No LTP data returned for token {instrument_token}. Response: {ltp_data}")
        except KiteException as e:
            self.logger.error(f"OrderPlacement: Kite API error fetching LTP for token {instrument_token}: {e}", exc_info=False) # exc_info=False to avoid too much noise for frequent calls
        except Exception as e:
            self.logger.error(f"OrderPlacement: General error fetching LTP for token {instrument_token}: {e}", exc_info=False)
        return None
    

class kiteAPIs:
    def __init__(self):
        self.Kite = None
        self.con = None
        self.startKiteSession = system_initialization()
        self.kite = self.startKiteSession.kite
        self.con = self.startKiteSession.con
        self.api_key = self.startKiteSession.api_key
        self.AccessToken = self.startKiteSession.GetAccessToken()
        self.kite.set_access_token(self.AccessToken)
        self.ticker = KiteTicker(self.startKiteSession.api_key, self.AccessToken)

        self.mysql_username = self.startKiteSession.mysql_username
        self.mysql_password = self.startKiteSession.mysql_password
        self.mysql_hostname = self.startKiteSession.mysql_hostname
        self.mysql_port = self.startKiteSession.mysql_port
        self.mysql_database_name = self.startKiteSession.mysql_database_name


    # getting the instrument token for a given symbol
    def get_instrument_token(self,symbol):
        query = f"SELECT instrument_token FROM kiteConnect.instruments_zerodha where tradingsymbol in ({symbol}) and instrument_type = 'EQ'"
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        # if len(df) > 0:
        #     return int(df.iloc[0,0])
        # else:
        #     return -1
        self.con.close()
        return df['instrument_token'].values.astype(int).tolist()
    # getting all the instrument tokens for a given instrument type
    def get_instrument_all_tokens(self, instrument_type):
        query = f"SELECT instrument_token FROM kiteConnect.instruments_zerodha where instrument_type = '{instrument_type}'".format(instrument_type)
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df['instrument_token'].values.astype(int).tolist()
    
    def get_instrument_active_tokens(self, instrument_type, end_dt_backfill):
        query = f"SELECT instrument_token FROM kiteConnect.instruments_zerodha where expiry >= '{end_dt_backfill}' and instrument_type = '{instrument_type}'".format(instrument_type, end_dt_backfill)
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df['instrument_token'].values.astype(int).tolist()

    def extract_data_from_db(self, from_date, to_date, interval, instrument_token):
        query = f"SELECT a.*, b.strike FROM kiteConnect.historical_data_{interval} a left join kiteConnect.instruments_zerodha b on a.instrument_token = b.instrument_token where date(a.timestamp) between '{from_date}' and '{to_date}' and a.instrument_token = {instrument_token}"
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df
    
    def convert_minute_data_interval(self, df, to_interval=1):
        if to_interval == 1:
            return df
        
        if df is None or df.empty:
            return pd.DataFrame() # Return empty DataFrame if input is empty

        if not isinstance(to_interval, int) or to_interval <= 0:
            raise ValueError("to_interval must be a positive integer.")

        # Ensure 'timestamp' column exists and is in datetime format
        # Assuming the datetime column is named 'timestamp' as per requirements.
        # If it's 'date' from getHistoricalData, it should be handled/renamed before this function
        # or this function should be adapted. For now, proceeding with 'timestamp'.
        if 'timestamp' not in df.columns:
            # Try to use 'date' column if 'timestamp' is missing, assuming it's the datetime column
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'}) 
            else:
                raise ValueError("DataFrame must contain a 'timestamp' or 'date' column for resampling.")
        
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            raise ValueError(f"Could not convert 'timestamp' column to datetime: {e}")

        # Sort by instrument_token and timestamp
        df = df.sort_values(by=['instrument_token', 'timestamp'])

        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'id': 'first', 
            'strike': 'first' 
        }

        all_resampled_dfs = []

        # Group by instrument_token and then by day for resampling
        # The pd.Grouper will use the 'timestamp' column, group by Day ('D'), using start_day as origin for daily grouping.
        grouped_by_token_day = df.groupby([
            'instrument_token', 
            pd.Grouper(key='timestamp', freq='D', origin='start_day')
        ])

        for (token, day_key), group_data in grouped_by_token_day:
            if group_data.empty:
                continue

            # Define the resampling origin for this specific day: 9:15 AM
            # day_key is the start of the day (00:00:00) from the Grouper
            origin_time_for_day = day_key + pd.Timedelta(hours=9, minutes=15)

            # Set timestamp as index for resampling this group
            group_data_indexed = group_data.set_index('timestamp')
            
            resampled_one_group = group_data_indexed.resample(
                rule=f'{to_interval}T',
                label='left', # Label of the interval is its start time
                origin=origin_time_for_day
            ).agg(agg_rules)

            # Drop rows where 'open' is NaN (meaning no data fell into this resampled interval)
            resampled_one_group = resampled_one_group.dropna(subset=['open'])

            if not resampled_one_group.empty:
                # Add instrument_token back as a column
                resampled_one_group['instrument_token'] = token
                all_resampled_dfs.append(resampled_one_group)
        
        if not all_resampled_dfs:
            return pd.DataFrame(columns=df.columns) # Return empty DF with original columns

        final_df = pd.concat(all_resampled_dfs)
        final_df = final_df.reset_index() # 'timestamp' becomes a column

        # Ensure final column order as per requirement
        # Desired order: ID, instrument_token, open, high, low, close, volume, strike, timestamp
        # Current columns likely: timestamp, open, high, low, close, volume, ID, strike, instrument_token
        
        # Define desired column order
        # (Make sure all these columns exist in final_df after aggregation and reset_index)
        # 'instrument_token' added above, 'timestamp' from reset_index
        desired_columns = ['ID', 'instrument_token', 'open', 'high', 'low', 'close', 'volume', 'strike', 'timestamp']
        
        # Filter out any columns that might not be present if original df was minimal
        # And reorder
        final_df_columns = [col for col in desired_columns if col in final_df.columns]
        final_df = final_df[final_df_columns]
        
        return final_df

    ## get data from kite API for a given token, from_date, to_date, interval

    def getHistoricalData(self, from_date, to_date, tokens, interval):
        # embed()
        if from_date > to_date:
            return
        
        if tokens == -1:
            print('Invalid symbol provided')
            return 'None'
        
        i = 0

        token_exceptions = []
        MAX_RETRIES = 3
        RETRY_DELAY_SECONDS = 5

        for t in tokens:
            print(f"Fetching data for token: {t}")
            records = None # Initialize records to None
            for attempt in range(MAX_RETRIES):
                try:
                    print(f"  Attempt {attempt + 1}/{MAX_RETRIES} for token {t}...")
                    records = self.kite.historical_data(t, from_date=from_date, to_date=to_date, interval=interval)
                    print(f"  Successfully fetched data for token {t} on attempt {attempt + 1}.")
                    break  # Success, exit retry loop
                except (KiteException, requests.exceptions.ReadTimeout, requests.exceptions.RequestException) as e:
                    print(f"  Error on attempt {attempt + 1} for token {t}: {type(e).__name__} - {str(e)}")
                    if attempt < MAX_RETRIES - 1:
                        print(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
                        time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        print(f"  Max retries reached for token {t}. Adding to exceptions.")
                        token_exceptions.append(t)
                        records = [] # Ensure records is an empty list on failure to avoid issues later
                except Exception as e: # Catch any other unexpected errors
                    print(f"  An unexpected error occurred on attempt {attempt + 1} for token {t}: {type(e).__name__} - {str(e)}")
                    if attempt < MAX_RETRIES - 1:
                        print(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
                        time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        print(f"  Max retries reached for token {t} due to unexpected error. Adding to exceptions.")
                        token_exceptions.append(t)
                        records = [] # Ensure records is an empty list
                        # Optionally re-raise the last exception if it's critical and unhandled by appending to token_exceptions
                        # raise
            
            # Continue with processing if records were fetched
            if records is not None and len(records) > 0:
                df = pd.DataFrame(records) # Convert to DataFrame here
                df['instrument_token'] = t
                # df = pd.concat([df, records_df], axis = 0) # This concat was for an older structure
                df['interval'] = interval
                df['id'] = df['instrument_token'].astype(str) + '_' + df['interval'] + '_' + pd.to_datetime(df['date']).dt.strftime("%Y%m%d%H%M") # Ensure date is used correctly
            
                self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
# Ensure 'date' column is pandas Timestamp for intermediate operations if not already
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])

                # Convert pandas Timestamp to string in 'YYYY-MM-DD HH:MM:SS' format for DB insertion
                # This matches how it would likely be formatted implicitly in row-by-row string insertion
                df['date_for_db'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

                # Prepare data for executemany: list of tuples
                # Column order matches the VALUES clause: id, instrument_token, date_for_db (string for timestamp col), open, high, low, close, volume
                try:
                    data_to_insert = list(df[['id', 'instrument_token', 'date_for_db', 'open', 'high', 'low', 'close', 'volume']].itertuples(index=False, name=None))
                except KeyError as ke:
                    print(f"DataFrame missing expected columns for DB insertion: {ke}. Columns available: {df.columns.tolist()}")
                    data_to_insert = [] 

                target_table_name = None
                if interval == 'minute':
                    target_table_name = "kiteConnect.historical_data_minute"
                elif interval == 'day':
                    target_table_name = "kiteConnect.historical_data_day"
                

                if target_table_name and data_to_insert:
                    cur = None # Initialize cur to None
                    try:
                        cur = self.con.cursor()
                        # Using %s placeholders for values in the query
                        query = f"INSERT IGNORE INTO {target_table_name} (id, instrument_token, timestamp, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                        # Assuming your table columns are named: id, instrument_token, timestamp (for date), open, high, low, close, volume
                        # If your table does not explicitly name columns in this order, adjust the query or ensure table was created with this order.
                        # For safety, it's best if INSERT specifies column names, like above.
                        # If you must stick to the original schemaless insert:
                        # query = f"INSERT IGNORE INTO {target_table_name} VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"


                        print(f"Attempting to bulk insert {len(data_to_insert)} records into {target_table_name}...")
                        cur.executemany(query, data_to_insert)
                        self.con.commit()
                        print(f"Bulk insert/ignore completed for {target_table_name}. Rows affected: {cur.rowcount}")
                    except Exception as e: # Catch potential errors during DB operation
                        print(f"Error during bulk insert into {target_table_name}: {e}")
                        # Consider self.con.rollback() if an error occurs and transactions are being used explicitly
                    finally:
                        if cur:
                            cur.close() # Close cursor
                            self.con.close() # Close connection after each token's DB operations
                elif not data_to_insert:
                    print(f"No data prepared for insertion for interval {interval} (data_to_insert list is empty).")
                else:
                    print(f"Unsupported interval '{interval}' or no table determined for database insertion.")
            elif records is not None and len(records) == 0: # Successfully fetched, but no data for the period
                 print(f"No historical data returned for token {t} for the given period.")
            # If records is None (all retries failed), it's already added to token_exceptions.
            
       
        print('token_exceptions: ',token_exceptions)
        # The original code returned 'df' which was the DataFrame for the *last* successfully processed token.
        # This might not be the desired behavior if processing multiple tokens.
        # If the goal is to collect all data, it should be aggregated.
        # For now, I'll keep it returning the last df, but this is a point of attention.
        # If all tokens fail, df might be uninitialized or from a much earlier success.
        # A safer return would be a list of dataframes or a concatenated one if all are successful.
        # Given the current structure, if all fail, df from previous loop iteration might be returned.
        # It's better to initialize df to an empty DataFrame at the start of the outer loop or handle returns more carefully.
        
        # Let's adjust to return an aggregated DataFrame or an empty one if all fail.
        all_data_dfs = []
        # The processing logic (DB insertion, etc.) should be inside the token loop if df is defined per token.
        # The current edit places the retry loop inside the token loop, and df is created from 'records'.
        # If the intention is to return a single DataFrame containing all data, then aggregation is needed.
        # The provided edit does not aggregate `df` across tokens. It processes one token at a time.
        # The `return df` at the end will return the DataFrame of the LAST token processed (or an error if it failed).
        # This seems consistent with the original structure where `df = pd.concat([df, records], axis = 0)` was commented out.
        # If you want to return *all* data fetched, we'd need to collect each token's df into a list and concat at the end.

        # For now, to ensure df is defined even if the last token fails but previous ones succeeded,
        # we should initialize df outside the loop, or adjust the return.
        # The current logic processes and potentially saves data per token. The final 'df' return is for the last token.
        # If the last token processing fails and `records` becomes `[]`, then `df` will be an empty DataFrame for that token.
        
        # The `df` variable is defined *inside* the `if records is not None and len(records) > 0:` block.
        # If the last token fails all retries, `records` will be `[]`, this block will be skipped,
        # and `df` might not be defined for the return if it was the *only* token.
        # Let's ensure df is initialized if we intend to always return it.
        
        # Given the original context, it seems `df` was intended to be the data for the current token being processed.
        # The return `df` at the very end is problematic if there are multiple tokens.
        # The primary action is saving to DB.
        # If a return value is truly needed for all data, a list of DFs or a concatenated DF should be built.
        # For now, I'll stick to the modified logic where `df` is per-token for DB saving and the final `return df`
        # will be the last token's data (or empty if it failed).
        # The `token_exceptions` list is the primary indicator of failures.

        # A small correction: Ensure 'df' is defined if the loop completes.
        # However, the current logic uses `df` within the loop for DB operations.
        # The final `return df` is likely a remnant or for a single-token use case.
        # If this function is always called with multiple tokens and an aggregated result is expected,
        # this return value needs rethinking. For now, I'll assume the DB saving is the main goal per token.

        # The simplest fix for the return value, if it must return *something* even if all fail,
        # is to initialize an empty df at the beginning of the function.
        # However, the loop structure processes one token at a time.
        
        # Let's assume the function's primary purpose is to fetch and store,
        # and the return value might be for convenience for the last token.
        # If 'records' is empty after retries, df won't be formed for that token.
        # The main thing is that `token_exceptions` tracks failures.
        # The print statements indicate progress.
        # The changes below focus on the retry logic as requested.
        final_df_to_return = pd.DataFrame() # Default empty return
        if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty: # If df was defined and has data from the last token
            final_df_to_return = df
        
        return final_df_to_return # Return df of the last processed token or empty if all failed / last one failed

    
