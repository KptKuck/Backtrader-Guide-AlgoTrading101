import finplot as fplt
import numpy as np
import pandas as pd
import yfinance as yf
from io import StringIO
from time import time

# load data and convert date
end_t = int(time())
start_t = end_t - 12*30*24*60*60 # twelve months
symbol = 'SPY'
interval = '1d'

def plot():
    ax, ax2 = fplt.create_plot('S&P 500 MACD', rows=2)

    return 1

def plot_dataset(data, symbol):

    df = data
    sy = symbol
    df['Date'] = pd.to_datetime(df['Date']).astype('int64') // 1_000_000  # use finplot's internal representation, which is ms

    ax = fplt.create_plot(sy, rows=1, maximize=False)


    # change to b/w coloring templates for next plots
    fplt.candle_bull_color = fplt.candle_bear_color = '#000'
    fplt.volume_bull_color = fplt.volume_bear_color = '#333'
    fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#fff'

    # plot price and volume
    fplt.candlestick_ochl(df[['Date', 'Open', 'Close', 'High', 'Low']], ax=ax)
    hover_label = fplt.add_legend('', ax=ax)
    axo = ax.overlay()
    fplt.volume_ocv(df[['Date', 'Open', 'Close', 'Volume']], ax=axo)
    fplt.plot(df.Volume.ewm(span=24).mean(), ax=axo, color=1)

    def update_legend_text(x, y):
        row = df.loc[df.Date == x]
        # format html with the candle and set legend
        fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.Open < row.Close).all() else 'a00')
        rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
        hover_label.setText(rawtxt % (symbol, interval.upper(), row.Open, row.Close, row.High, row.Low))

    def update_crosshair_text(x, y, xtext, ytext):
        ytext = '%s (Close%+.2f)' % (ytext, (y - df.iloc[x].Close))
        return xtext, ytext

    fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
    fplt.add_crosshair_info(update_crosshair_text, ax=ax)

    fplt.show()


    return 1


