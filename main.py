import mplfinance as mpf
import streamlit as st
import yfinance as yf

from src.agents import financial_agent, social_agent

st.set_page_config(layout="wide")


def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return not data.empty
    except Exception:
        return False


def company_name_n_currency(ticker):
    stock = yf.Ticker(ticker)
    company = stock.info.get("longName", "")
    currency = stock.info.get("currency", "")
    return company, currency


left, right = st.columns([0.35, 0.65], vertical_alignment="top")

with left:
    st.title("Write a valid ticker")
    col1, col2 = st.columns([0.7, 0.3], vertical_alignment="bottom")
    with col1:
        ticker = st.text_input(
            "Ticker",
            value="EQNR",
        )
    with col2:
        analyze = st.button("**Analyze!**", type="primary", use_container_width=True)

    if analyze and ticker:
        company, currency = company_name_n_currency(ticker)

        results = financial_agent.query_financial_agent(ticker, company)
        st.header("Financial News Analysis")
        st.subheader("Query")
        st.text(results.query)
        summary = results.summary
        if not summary or "empty" in summary:
            st.text("No financial news was retrieved for the ticker.")
        else:
            st.text(summary)
            st.subheader("Sentiment")
            st.text(results.sentiment)

        results = social_agent.query_financial_agent(ticker, company)
        st.header("Social Analysis")
        st.subheader("Query")
        st.text(results.query)
        st.subheader("Summary")
        summary = results.summary
        if not summary or "empty" in summary:
            st.text("No social info was retrieved for the ticker.")
        else:
            st.text(summary)
            st.subheader("Sentiment")
            st.text(results.sentiment)


with right:
    st.title("Ploting Market data")
    if analyze and ticker:
        try:
            data = yf.download(
                ticker, period="1mo", interval="1h", multi_level_index=False
            )

            # Calculate RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            rs = gain / loss
            data["RSI"] = 100 - (100 / (1 + rs))
            # Compute moving average and standard deviation
            data["Middle_Band"] = data["Close"].rolling(window=20).mean()
            data["Upper_Band"] = data["Middle_Band"] + (
                data["Close"].rolling(window=20).std() * 2
            )
            data["Lower_Band"] = data["Middle_Band"] - (
                data["Close"].rolling(window=20).std() * 2
            )
            # Create additional plots
            apdict = [
                mpf.make_addplot(data["Upper_Band"], color="teal", alpha=0.7),
                mpf.make_addplot(data["Middle_Band"], color="dodgerblue", alpha=0.7),
                mpf.make_addplot(data["Lower_Band"], color="lightseagreen", alpha=0.7),
                mpf.make_addplot(
                    data["RSI"], panel=1, ylabel="RSI", color="purple", alpha=0.6
                ),
                mpf.make_addplot(
                    [70] * len(data), panel=1, color="purple", linestyle="dashed"
                ),
                mpf.make_addplot(
                    [30] * len(data), panel=1, color="purple", linestyle="dashed"
                ),
            ]

            fig, ax = mpf.plot(
                data,
                type="candle",
                style="binance",
                addplot=apdict,
                title="Candlestick with RSI on Subplot",
                ylabel=f"Price ({currency})",
                returnfig=True,
            )
            st.pyplot(fig)

        except Exception:
            st.write("Problem loading market data")
