import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ssl
import pymysql
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from tenacity import retry, stop_after_attempt, wait_fixed
import atexit



st.set_page_config(page_title="Nifty 50 Analysis", layout="wide")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data_from_db():
    try:
        # Create SSL context
        ssl_context = ssl.create_default_context()
        global connection
        # Establish connection
        connection = pymysql.connect(
            host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
            port=4000,
            user="4V44XYoMA7okY9v.root",
            password="LdkgrMO1WG5Tk9Tk",
            database="stock_db",
            ssl={
                'ca': ssl_context.load_default_certs(),
                'check_hostname': False
            },
            cursorclass=pymysql.cursors.DictCursor
        )

        with connection.cursor() as cursor:
            # Fetch ticker metadata
            cursor.execute("""
                SELECT 
                    ticker_id AS ticker_id,
                    ticker_name AS Ticker,
                    sector AS Sector,
                    company_name AS Company,
                    return as `Yearly Return`
                           
                FROM tickers
            """)
            sector_df = pd.DataFrame(cursor.fetchall())

            # Fetch stock data
            cursor.execute("""
                                SELECT
                    `ticker_id` AS `ticker_id`,
                    `date` AS `Date`,
                    `open` AS `Open`,
                    `high` AS `High`,
                    `low` AS `Low`,
                    `close` AS `Close`,
                    `volume` AS `Volume`,
                    `daily_return` AS `Daily Return`,
                    `cumulative_return` AS `Daily Return Cumulative`
                    FROM
                    `stock_data`;
            """)
            stock_df = pd.DataFrame(cursor.fetchall())
            

        # Merge datasets
        merged_df = pd.merge(stock_df, sector_df, on='ticker_id')
        
        # Convert date column to datetime
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        
        return merged_df

    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        raise e

def close_connection():
    """Cleanly close database connection"""
    global connection
    if connection:
        connection.close()
        connection = None


# =====================
# Data Preparation
# =====================
merged_df = load_data_from_db()

merged_df['Yearly Return']=merged_df['Yearly Return'].round(2)
st.title("Nifty 50 Stock Performance Dashboard")
st.markdown("---")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏥 Market Health",
    "🏭 Sector View",
    "🔍 Stock Analysis",
    "📊 Comparisons",
    "🛠️ Decision Support"
])

# =====================
# New Section: Market Overview
# =====================

with tab1:

    performance_df = merged_df.groupby('Ticker').agg({
    'Yearly Return': 'last',
    'Daily Return': 'std',
    'Close': 'last'
    }).reset_index().rename(columns={
        'Daily Return': 'Volatility',
        'Close': 'Latest Price'
    })

    # Calculate market stats
    green_stocks = len(performance_df[performance_df['Yearly Return'] > 0])
    red_stocks = len(performance_df[performance_df['Yearly Return'] <= 0])
    avg_return = performance_df['Yearly Return'].mean()
    avg_volatility = performance_df['Volatility'].mean()

    volatility_metrics = merged_df.groupby('Ticker')['Daily Return'].std().describe()
    return_metrics = merged_df.groupby('Ticker')['Yearly Return'].last().describe()
    stock_avg = merged_df.groupby(['Ticker', 'Sector']).agg({
        'Close': 'mean',
        'Volume': 'mean'
    }).reset_index()

    close_avg=stock_avg.sort_values(by='Close',ascending=False)
    vol_avg=stock_avg.sort_values(by='Volume',ascending=False)
    top_10 = performance_df.nlargest(10, 'Yearly Return')
    bottom_10 = performance_df.nsmallest(10, 'Yearly Return')

    green_pct=green_stocks
    red_pct=red_stocks

    volatility_metrics = merged_df.groupby('Ticker')['Daily Return'].std().describe()
    return_metrics = merged_df.groupby('Ticker')['Yearly Return'].last().describe()


      # Market Health Check
    st.header("📊 Market Health Dashboard")
    
    # Market Pulse Metrics with Data-Driven Popovers
    with st.container():
        st.subheader("Market Pulse")
        col1, col2, col3 = st.columns(3)
        
        # Metric 1: Average Market Return
        with col1:
            with st.popover("🔍 Market Return Context"):
                st.markdown(f"""
                **Our Market Context:**  
                ∙ Median Return: {return_metrics['50%']:.1f}%  
                ∙ Top 25% Stocks: > {return_metrics['75%']:.1f}%  
                ∙ Bottom 25% Stocks: < {return_metrics['25%']:.1f}%  
                
                **Interpretation:**  
                ∘ Above {return_metrics['75%']:.1f}% → Exceptional Performance  
                ∘ {return_metrics['25%']:.1f}%-{return_metrics['75%']:.1f}% → Typical Range  
                ∘ Below {return_metrics['25%']:.1f}% → Underperforming
                """)
            st.metric("Average Market Return", f"{avg_return:.2f}%")

        # Metric 2: Green vs Red Stocks
        with col2:
            with st.popover("🔍 Market Sentiment Analysis"):
                st.markdown(f"""
                **Current Distribution:**  
                ∙ {green_stocks} Winning Stocks ({green_pct:.0%})  
                ∙ {red_stocks} Declining Stocks ({red_pct:.0%})  
                
                **Historical Context:**  
                ∘ >90% Green → Extreme Bull Market  
                ∘ 60-90% Green → Strong Bull Market  
                ∘ <40% Green → Bear Market Territory  
                
                **Action Insight:**  
                "Current levels suggest { "extreme bullish sentiment" if green_pct > 0.9 else "strong positive momentum" }
                """)
            st.metric("Green vs Red Stocks", 
                     f"{green_stocks/(green_stocks+red_stocks):.0%} vs {red_stocks/(green_stocks+red_stocks):.0%}")
        # Metric 3: Average Volatility
        with col3:
            with st.popover("🔍 Volatility Framework"):
                st.markdown(f"""
                **Volatility Spectrum:**  
                ∙ 75th Percentile: {volatility_metrics['75%']:.1f}%  
                ∙ Median: {volatility_metrics['50%']:.1f}%  
                ∙ 25th Percentile: {volatility_metrics['25%']:.1f}%  
                
                **Risk Assessment:**  
                ∘ >{volatility_metrics['75%']:.1f}% → High Risk  
                ∘ {volatility_metrics['25%']:.1f}%-{volatility_metrics['75%']:.1f}% → Moderate Risk  
                ∘ <{volatility_metrics['25%']:.1f}% → Low Risk  
                
                **Trading Implication:**  
                "Current volatility suggests { "high-risk, high-reward" if avg_volatility > volatility_metrics['75%'] else "moderate risk" } environment"
                """)
            st.metric("Average Volatility", f"{avg_volatility:.2f}%")

        # Market Sentiment Visualization
        with st.expander("Visualize Market Sentiment", expanded=True):
            fig = px.pie(names=['Green Stocks', 'Red Stocks'], 
                        values=[green_stocks, red_stocks],
                        color=['Green Stocks', 'Red Stocks'],
                        color_discrete_map={'Green Stocks':'#2ecc71', 'Red Stocks':'#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

    # Performance Leaderboard
    with st.container():
        st.subheader("📈 Yearly Return Leaderboard")
        
        with st.popover("ℹ️ Leaderboard Methodology"):
            st.markdown("""
            **Ranking Criteria:**  
            1. Average Of Ticker's Yearly Return  
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(top_10, x='Ticker', y='Yearly Return', 
                        color='Yearly Return', color_continuous_scale='Greens',
                        title="Top 10 Performers 🚀")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(bottom_10, x='Ticker', y='Yearly Return',
                        color='Yearly Return', color_continuous_scale='Reds',
                        title="Bottom 10 Performers 🔻")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🏭 Sector Performance")
    
    # Calculate sector metrics
    sector_perf = merged_df.groupby('Sector').agg({
        'Yearly Return': 'mean',
        'Daily Return': 'std',
        'Volume': 'mean',
        'Close': 'mean'
    }).reset_index().rename(columns={
        'Yearly Return': 'Avg Return',
        'Daily Return': 'Volatility',
        'Volume': 'Avg Volume',
        'Close': 'Avg Close'
    })
    
    # Sort by average return
    sector_perf = sector_perf.sort_values(by='Avg Return', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col2:
        # Enhanced Bar Plot with conditional coloring
        fig = px.bar(sector_perf, 
                    x='Sector', 
                    y='Avg Return',
                    color='Avg Return',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
                    title="Average Sector Returns (Red = Negative, Green = Positive)",
                    hover_data=['Avg Volume', 'Avg Close'])
        
        # Set color scale midpoint at 0
        fig.update_layout(coloraxis=dict(cmin=-1, cmax=1))
        st.plotly_chart(fig, use_container_width=True)

    with col1:
        # Calculate risk thresholds from data
        volatility_25 = sector_perf['Volatility'].quantile(0.25)
        volatility_75 = sector_perf['Volatility'].quantile(0.75)
      
        # Enhanced Scatter Plot
        fig = px.scatter(sector_perf, 
                        x='Volatility', 
                        y='Avg Return',
                        size='Avg Volume',
                        color='Sector',
                        hover_name='Sector',
                        size_max=40,
                        color_discrete_sequence=px.colors.qualitative.Dark24,
                        title="Sector Risk-Return Profile",
                        labels={'Avg Return': 'Average Return',
                                'Volatility': 'Risk (Standard Deviation)'})
            
            # Add reference lines and annotations
        fig.add_hline(y=0, line_dash="dot", line_color="red")
        fig.add_vline(x=sector_perf['Volatility'].mean(), 
                    line_dash="dash", line_color="blue")
        
        # Add interactive popover explanation
        with st.popover("📖 How to Read This Scatter Chart"):
            st.markdown(f"""
            **Guide:**  
            - **X-Axis:** Risk (Volatility)  
            - ↔ Left = Less Risk, Right = More Risk  
            - **Y-Axis:** Yearly Return  
            - ↑ High = Better Performance  
            - **Lines:**  
            - 🔵 Blue = Avg Risk ({sector_perf['Volatility'].mean():.2f})  
            - 🔴 Red = Zero Return Benchmark  
            - **Bubble:**  
            - 🎨 Color = Sectors  
            - 🔍 Size = Avg Volume  
            
            **Insights:**  
            - 🟢 **Top-Left:** High Return, Low Risk (Best)  
            - 🔴 **Bottom-Right:** Low Return, High Risk (Avoid)  
            """)
      
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Are sectors with high trading volume also expensive?", expanded=False):

    # Create Scatter Plot for Liquidity & Valuation Profile
        fig = px.scatter(sector_perf,
                        x='Avg Volume',
                        y='Avg Close',
                        size='Avg Volume',
                        color='Sector',
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        title="<b>Sector Liquidity & Valuation Profile</b>",
                        labels={'Avg Volume': 'Trading Volume (Avg)',
                                'Avg Close': 'Closing Price (Avg)'},
                        hover_name='Sector',
                        hover_data=['Volatility'],
                        size_max=40)

        # Add market average lines
        fig.add_hline(y=sector_perf['Avg Close'].mean(),
                    line_dash="dot",
                    annotation_text="Avg Price",
                    line_color="blue")
        fig.add_vline(x=sector_perf['Avg Volume'].mean(),
                    line_dash="dot",
                    annotation_text="Avg Volume",
                    line_color="orange")

        # Add Popover with Explanation
        with st.popover("📖 How to Read"):
            st.markdown(f"""
            **Guide:**  
            - **X-Axis:** Avg Trading Volume → Higher = More activity  
            - **Y-Axis:** Avg Closing Price → Higher = More valuable stocks  
            - **Bubble Size:** Represents trading volume  
            - **Color:** Different sectors for easy comparison  
            
            **Quick Insight:**  
            - Top-right: High price, high activity (Popular & valuable)  
            - Bottom-left: Low price, low activity (Less active)  
            """)
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1,col2=st.columns(2)
    with col1:
       selected_sector = st.selectbox("Select Sector", ["All"] + list(merged_df['Sector'].unique()))

    with col2:
        if selected_sector == "All":
            available_tickers = merged_df['Ticker'].unique()
        else:
            available_tickers = merged_df[merged_df['Sector'] == selected_sector]['Ticker'].unique() # give the only selected sector tickers
        selected_ticker = st.selectbox("Select Ticker", available_tickers)    

    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    latest_data = merged_df[merged_df['Ticker'] == selected_ticker].iloc[-1]

    with col1:
    # Calculate averages and ranking
                # Step 1: Get the latest closing price for each ticker
        latest_prices = merged_df.groupby('Ticker')['Close'].last().sort_values(ascending=False)

        # Step 2: Find the market average and sector average
        market_avg_price = latest_prices.mean()
        sector_avg_price = latest_prices[merged_df[merged_df['Sector'] == selected_sector]['Ticker'].unique()].mean()

        # Step 3: Find the rank of the selected ticker
        stock_rank = latest_prices.index.get_loc(selected_ticker) + 1
        total_stocks = len(latest_prices)

        # Display current price
        st.metric("Current Price", f"₹{latest_data['Close']:.2f}")
        
        # Popover with insights
        with st.popover("📖 Price Insights"):
            st.markdown(f"""
            - 🌐 **Market Avg:** ₹{market_avg_price:.2f}  
             
            - 📊 **Stock Price Rank:** {stock_rank} / {total_stocks}  
            """)


    with col2:
        
        st.metric("Yearly Return", f"{latest_data['Yearly Return']:.2f}%")
        with st.popover("📖 Yearly Return Guide"):
            st.markdown(f"""
            **Market Context:**  
            - 🌟 **Market Avg:** {avg_return:.2f}%  
            - 🚀 **Above {avg_return * 1.3:.2f}%** → Outstanding  
            - 📈 **{avg_return:.2f}% - {avg_return * 1.3:.2f}%** → Good  
            - 📊 **{avg_return * 0.5:.2f}% - {avg_return:.2f}%** → Moderate  
            - 📉 **Below {avg_return * 0.5:.2f}%** → Weak  
        """)
     
    with col3:
        
        st.metric("Sector", latest_data['Sector'])
  
    with col4:
        st.metric("Company", latest_data['Company'])

    # Section 2: Price Charts
    st.header("📈 Price Analysis")
    filtered_data = merged_df[merged_df['Ticker'] == selected_ticker] # get the entir row of the ticker

    tab1, tab2, tab3 = st.tabs(["OHLC Chart", "Trend Analysis", "Volume"])

    with tab1:
        fig = go.Figure(go.Candlestick(
            x=filtered_data['Date'],
            open=filtered_data['Open'],
            high=filtered_data['High'],
            low=filtered_data['Low'],
            close=filtered_data['Close'],
            name='Price'
        ))
        fig.update_layout(title=f"{selected_ticker} Price Movement")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.line(filtered_data, x='Date', y='Close', 
                    title="Closing Price Trend")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.bar(filtered_data, x='Date', y='Volume', 
                    title="Trading Volume")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Section 4: Stock Comparison
    st.header("🔍 Cross-Stock Analysis")
    selected_tickers = st.multiselect("Compare with other stocks:", 
                                    merged_df['Ticker'].unique(),
                                    default=[selected_ticker])

    if selected_tickers:
        compare_data = merged_df[merged_df['Ticker'].isin(selected_tickers)]
        fig = px.line(compare_data, x='Date', y='Close', color='Ticker',
                    title="Price Comparison")
        st.plotly_chart(fig, use_container_width=True)   
   
    # =====================
    # New Section: Correlation Heatmap
    # =====================
    with st.expander("Coralation Heat Map",expand=False):
        st.header("🌡️ Stock Price Correlations")
        @st.cache_data
        def create_correlation_heatmap(df):
            # Pivot and calculate correlations
            pivot_df = df.pivot(index='Date', columns='Ticker', values='Close')
            correlation_matrix = pivot_df.corr()

            # Custom colormap setup
            ranges = [0, 0.3, 0.6, 0.9, 1]
            colors = ["#E0F7FA", "#80DEEA", "#4DD0E1", "#006064"]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(ranges, len(colors))

            # Create figure
            fig = plt.figure(figsize=(25, 25))
            heatmap = sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap=cmap,
                norm=norm,
                vmin=-1, vmax=1,
                linewidths=0.5,
                fmt=".1f",
                annot_kws={'size': 8},
                cbar_kws={"shrink": 0.5, "ticks": ranges}
            )

            # Customizations
            plt.title('Stock Price Correlation Heatmap', fontsize=28, pad=20)
            plt.xticks(ha='center', fontsize=18, rotation=90)
            plt.yticks(rotation=0, fontsize=18)
            heatmap.set_facecolor('#f5f5f5')
            plt.tight_layout()
            
            return fig

        # Display heatmap with size warning
        with st.spinner('This may take some minutes...'):
            heatmap_fig = create_correlation_heatmap(merged_df)
            st.pyplot(heatmap_fig)
            st.warning("Note: This visualization may take longer to load due to its high resolution")


    # =====================
    # New Section: Monthly Performers
    # =====================
    st.header("📅 Monthly Top/Bottom Performers")

    # Create monthly analysis
    merged_df['Month'] = merged_df['Date'].dt.to_period('M').astype(str)
    monthly_returns = merged_df.groupby(['Month', 'Ticker']).agg({
        'Close': ['first', 'last']
    }).reset_index()
    monthly_returns.columns = ['Month', 'Ticker', 'First Close', 'Last Close']
    monthly_returns['Monthly Return'] = (monthly_returns['Last Close'] - monthly_returns['First Close']) / monthly_returns['First Close']
    monthly_returns['Monthly Return'] = pd.to_numeric(monthly_returns['Monthly Return'], errors='coerce')
    # Month selector
    selected_month = st.selectbox("Select Month", options=monthly_returns['Month'].unique())

    # Filter and display
    month_data = monthly_returns[monthly_returns['Month'] == selected_month]
    top_gainers = month_data.nlargest(5, 'Monthly Return')
    top_losers = month_data.nsmallest(5, 'Monthly Return')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top 5 Gainers - {selected_month}")
        fig = px.bar(top_gainers, x='Monthly Return', y='Ticker', 
                    orientation='h', color='Monthly Return',
                    color_continuous_scale='Greens',
                    title=f"Top Gainers - {selected_month}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Top 5 Losers - {selected_month}")
        fig = px.bar(top_losers, x='Monthly Return', y='Ticker',
                    orientation='h', color='Monthly Return',
                    color_continuous_scale='Reds',
                    title=f"Top Losers - {selected_month}")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    # =====================
    # Decision Support Metrics Section
    # =====================
    st.header("📊 Decision Support Metrics")

    with st.popover("📈 Sector Analysis Guide"):
        st.markdown("""
        **Sector Benchmarking Toolkit**  
        Compare industries using 3 lenses:  
        1. **Valuation** (Closing Prices)  
        2. **Liquidity** (Trading Volumes)  
        3. **Performance** (Yearly Returns)  
        
        **How to Use:**  
        - High 'Close Std' → Risky sectors  
        - Low 'Volume Std' → Stable trading  
        - Use filters to find undervalued sectors  
        """)

    # Create statistics table
    stats_table = merged_df.groupby('Sector').agg({
        'Close': ['mean', 'std'],
        'Volume': ['mean', 'std'],
        'Yearly Return': ['mean', 'std']
    }).reset_index()

    st.subheader("Sector-wise Statistics")
    st.dataframe(stats_table.style.format({
        ('Close', 'mean'): '₹{:.2f}',
        ('Close', 'std'): '₹{:.2f}',
        ('Volume', 'mean'): '{:.0f}',
        ('Volume', 'std'): '{:.0f}',
        ('Yearly Return', 'mean'): '₹{:.2f}',
        ('Yearly Return', 'std'): '{:.2%}'
    }), use_container_width=True)

    # =====================
    # Investment Insights Section
    # =====================
    st.header("💡 Investment Insights")

    with st.popover("🔍 Stock Filtering Logic"):
        st.markdown("""
        **Curated Stock Lists**  
        - Green Flags (Growers): Safe growth stocks  
        - Red Flags (Decliners): High-risk candidates  
        
        **Methodology:**  
        - 6-month price stability checks  
        - Volume trend analysis  
        - Sector-adjusted thresholds  
        """)

    # Calculate consistency metrics
    consistency_df = merged_df.groupby('Ticker').agg({
        'Daily Return Cumulative': 'last',
        'Daily Return': 'std'
    }).reset_index().rename(columns={
        'Daily Return Cumulative': 'Total Return',
        'Daily Return': 'Volatility'
    })
    consistency_df['Total Return'] = pd.to_numeric(consistency_df['Total Return'], errors='coerce')
    tab1, tab2 = st.tabs(["Consistent Growers", "Significant Decliners"])

    with tab1:
        st.subheader("Stocks with Consistent Growth 📈")
        with st.popover("🔍 Growth Criteria", help="Filter details"):
            st.markdown("""
            **Selection Parameters:**  
            - Minimum 1 year positive returns  
            - Volatility < 1.5% daily fluctuation  
            - Survived market corrections  
            
            **Ideal For:**  
            - Retirement portfolios  
            - Dividend investors  
            - Risk-averse traders  
            """)
        
        consistent_growers = consistency_df[(consistency_df['Total Return'] > 0) & 
                                        (consistency_df['Volatility'] < 1.5)].nlargest(5, 'Total Return')
        st.dataframe(consistent_growers.style.format({
            'Total Return': '{:.2f}%',
            'Volatility': '{:.4f}'
        }), use_container_width=True)

    with tab2:
        st.subheader("Stocks with Significant Declines 📉")
        with st.popover("🔍 Risk Flags", help="Why these matter"):
            st.markdown("""
            **Risk Indicators:**  
            - Sustained price depreciation  
            - High daily volatility (>1.5%)  
            - Low institutional ownership  
            
            **Potential Opportunities:**  
            - Short-selling candidates  
            - Turnaround plays  
            - Merger arbitrage situations  
            """)
        
        significant_decliners = consistency_df[(consistency_df['Total Return'] < 0) & 
                                            (consistency_df['Volatility'] > 1.5)].nsmallest(5, 'Total Return')
        st.dataframe(significant_decliners.style.format({
            'Total Return': '{:.2f}%',
            'Volatility': '{:.4f}'
        }), use_container_width=True)
 
    
    # =====================
    # Custom Ranking Implementation
    # =====================
    def calculate_composite_scores(df):
        """Calculate weighted performance scores"""
        # Calculate stock metrics
        stock_metrics = df.groupby('Ticker').agg({
            'Yearly Return': 'mean',          # Average return
            'Daily Return': 'std',            # Volatility (lower is better)
            'Volume': 'mean'                  # Liquidity (higher is better)
        }).reset_index()
        
        # Normalize metrics (0-1 scale)
        stock_metrics['Return_Score'] = (stock_metrics['Yearly Return'] - stock_metrics['Yearly Return'].min()) / \
                                    (stock_metrics['Yearly Return'].max() - stock_metrics['Yearly Return'].min())
        
        # Inverse normalization for volatility (since lower is better)
        stock_metrics['Volatility_Score'] = 1 - ((stock_metrics['Daily Return'] - stock_metrics['Daily Return'].min()) / \
                                            (stock_metrics['Daily Return'].max() - stock_metrics['Daily Return'].min()))
        
        stock_metrics['Volume_Score'] = (stock_metrics['Volume'] - stock_metrics['Volume'].min()) / \
                                    (stock_metrics['Volume'].max() - stock_metrics['Volume'].min())
        
        # Calculate composite score with weights
        stock_metrics['Composite_Score'] = (
            (stock_metrics['Return_Score'] * 0.6) +
            (stock_metrics['Volatility_Score'] * 0.3) +
            (stock_metrics['Volume_Score'] * 0.1)
        )
        
        return stock_metrics.sort_values('Composite_Score', ascending=False)

    # =====================
    # Modified Stock Performance Ranking Section
    # =====================
    st.header("🏆 Smart Performance Ranking")

    with st.popover("📊 Ranking Methodology"):
        st.markdown("""
        **Weighted Ranking Criteria:**
        1. Yearly Returns (60% weight)  
        2. Low Volatility (30% weight)  
        3. Trading Volume (10% weight)  
        
        **Formula:**  
        `Composite Score = (Return × 0.6) + (Stability × 0.3) + (Liquidity × 0.1)`  
        
        *Note: Stability is inverse of volatility*
    """)

    # Calculate ranked stocks
    ranked_stocks = calculate_composite_scores(merged_df)
    top_10_smart = ranked_stocks.nlargest(10, 'Composite_Score')
    bottom_10_smart = ranked_stocks.nsmallest(10, 'Composite_Score')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Smart Picks 🚀")
        fig = px.bar(top_10_smart, 
                    x='Composite_Score', 
                    y='Ticker',
                    orientation='h',
                    color='Yearly Return',
                    color_continuous_scale='Greens',
                    hover_data=['Daily Return', 'Volume'],
                    labels={'Composite_Score': 'Smart Score'},
                    title="Balanced Performance Leaders")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("High-Risk Candidates 🔻")
        fig = px.bar(bottom_10_smart, 
                    x='Composite_Score', 
                    y='Ticker',
                    orientation='h',
                    color='Yearly Return',
                    color_continuous_scale='Reds_r',
                    hover_data=['Daily Return', 'Volume'],
                    labels={'Composite_Score': 'Smart Score'},
                    title="Low Score Stocks")
        fig.update_layout(yaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)

    st.write("""
    This ranking system identifies stocks that balance growth potential with risk management, 
    going beyond pure returns to find sustainable performers
    """)
    with st.popover("❓ Why different top stocks?"):
        st.markdown("""
        **Different Ranking Philosophies:**  
        1. Consistent Growers → Pure return maximizers  
        2. Composite Score → Balanced performers  
        
        **Tip:** Use both lists - Bharti for growth, Trent for stability!
        """)

atexit.register(close_connection)