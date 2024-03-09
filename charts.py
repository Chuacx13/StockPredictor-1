
from plotly import graph_objs as go

def plot_open_raw_data(data, st):
    load = st.text("Loading graph...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open', hovertemplate='Open(%{x|%Y-%m-%d}): %{y}'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    load.text("")


def plot_close_raw_data(data, st):
    load = st.text("Loading graph...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close', hovertemplate='Open(%{x|%Y-%m-%d}): %{y}'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    load.text("")
