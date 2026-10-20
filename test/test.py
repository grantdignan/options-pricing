import pandas as pd
from options_pricing.pricing import parity

calls = pd.read_csv('test/calls.csv')
taus = calls['tau'].to_numpy()
calls = calls.drop(columns='tau')
strikes_calls = calls.columns.to_numpy(dtype=float)
calls = calls.values

puts = pd.read_csv('test/puts.csv')
puts = puts.drop(columns='tau')
puts.to_csv('test/puts1.csv', index=False)
strikes_puts = puts.columns.to_numpy(dtype=float)
puts = puts.values

rates = pd.read_csv('test/rates.csv')['r'].to_numpy()[:len(taus)]

calls_parity = parity.calls(puts, strikes_puts, taus, rates, 7690.73)
print(calls_parity)