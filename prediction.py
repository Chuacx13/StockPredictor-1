import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import date

START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


