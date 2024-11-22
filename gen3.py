import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import yfinance as yf
import streamlit as st

# Set up OpenAI API key
openai_api_key = "sk-SEbL52G-t-KNL6SEypzKn9XxrNZ73TMFITlhBfAiXuT3BlbkFJ6aE0EtEyQQXoNcWMMS1HBmS6gtSQb5w3MEk1z1EMkA"  # Replace with your actual OpenAI API key

# Define LangChain prompt template
prompt_template = PromptTemplate(
    input_variables=["stock_name", "period", "recommendation"],
    template="""Summarize the recent financial performance of {stock_name} for the period {period}.
    Include key insights, challenges faced, and the future outlook. 
    Based on the performance, please suggest if it's a good time to buy the stock, and if so, why or why not.
    {recommendation}"""
)

# Initialize LangChain LLM with the API key passed explicitly
llm = ChatOpenAI(temperature=0.7, model="gpt-4", openai_api_key=openai_api_key)

# Define a chain for summarization
summary_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit App
st.title("AI-Powered Financial Insight Generator")
st.markdown(
    """
    This tool provides AI-generated financial insights based on recent stock performance.
    - **Enter a stock ticker** (e.g., AAPL for Apple, TSLA for Tesla).
    - **Specify a period** (e.g., "last quarter").
    """
)

# User inputs
stock_name = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", placeholder="Stock ticker symbol")
period = st.text_input("Enter analysis period (e.g., last quarter):", placeholder="Timeframe for analysis")

# Generate insights
if st.button("Generate Insights"):
    with st.spinner("Fetching data and generating insights..."):
        try:
            # Fetch stock data using Yahoo Finance
            ticker = yf.Ticker(stock_name)
            stock_data = ticker.history(period="6mo")

            # Validate if data is available
            if stock_data.empty:
                st.error("No data found for the given stock ticker. Please try another ticker.")
            else:
                st.write(f"Stock data for **{stock_name}**:")

                # Display the stock data in a table format
                st.dataframe(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])

                # Calculate Simple Moving Averages (SMA) for buy signal (e.g., 50-day and 200-day SMA)
                stock_data['50_SMA'] = stock_data['Close'].rolling(window=50).mean()
                stock_data['200_SMA'] = stock_data['Close'].rolling(window=200).mean()

                # Display moving averages
                st.subheader("Moving Averages (50-day & 200-day)")
                st.line_chart(stock_data[['Close', '50_SMA', '200_SMA']])

                # Suggestion logic based on moving averages
                recommendation = ""
                if stock_data['50_SMA'][-1] > stock_data['200_SMA'][-1]:
                    recommendation = "The stock is currently in an uptrend, and it may be a good time to buy based on the moving averages."
                else:
                    recommendation = "The stock is currently in a downtrend, and it might be better to wait before buying."

                # Generate insights using LangChain with buy recommendation
                result = summary_chain.run(stock_name=stock_name.upper(), period=period, recommendation=recommendation)
                st.subheader("AI-Generated Financial Insights & Buy Recommendation:")
                st.write(result)

                # Optional: Display recent headlines
                st.subheader("Recent News Headlines:")
                try:
                    news = ticker.news
                    for article in news[:5]:  # Show top 5 news items
                        st.markdown(f"- [{article['title']}]({article['link']})")
                except Exception:
                    st.write("Unable to fetch news at this time.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Powered by OpenAI and LangChain | Built with 💡 by Generative AI.")
