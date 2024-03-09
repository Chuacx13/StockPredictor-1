from plotly import graph_objs as go

def plot_open_raw_data(data, st):
    load = st.text("Loading graph...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                              y=data['Open'], 
                              name='stock_open', 
                              hovertemplate='Open(%{x|%Y-%m-%d}): %{y}'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    load.text("")

def plot_close_raw_data(data, st):
    load = st.text("Loading graph...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['Close'], 
                             name='stock_close', 
                             hovertemplate='Open(%{x|%Y-%m-%d}): %{y}'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    load.text("")

def plot_hold_vs_trade(df, st):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Trade'], 
                             mode='lines', 
                             name='Signal based strategy', 
                             line=dict(color='green')))
    
    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Hold'], 
                             mode='lines', 
                             name='Buy and Hold strategy', 
                             line=dict(color='red')))
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Wealth', legend_title='Type of Strategy')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_profit_loss(df, st):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Profit'], 
                             mode='lines'))
    fig.add_hline(y=0, line=dict(color='red', width=1))
    fig.update_layout(xaxis_title='Date', yaxis_title='Profit/Loss')
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
