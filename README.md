# Stock Analysis

The Stock Performance Dashboard aims to provide a comprehensive visualization and analysis of the Nifty 50 stocks' performance over the past year, helping investors, analysts, and enthusiasts make informed decisions based on stock performance trends.

# Raw data sample
   - Ticker: SBIN
   - close: 602.95
   - date: '2023-10-03 05:30:00'
   - high: 604.9
   - low: 589.6
   - month: 2023-10
   - open: 596.6
   - volume: 15322196



## Features
- Convert YAML data to CSV format.
- Clean and manipulate stock data using Python scripts.
- Perform data analysis and generate insights.
- Create interactive dashboards using Power BI.
- Execute SQL queries for data exploration and analysis.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Adchayakumar/Stock-Analysis.git
   ```
2. Navigate to the project directory:
   ```
   cd Stock-Analysis
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the main Python script:
   ```
   python main.py
   ```

## Usage
1. Place your stock data files in the `data/` folder.
2. Convert YAML data to CSV:
   ```
   python yaml_to_csv.py
   ```
3. Open the Power BI dashboard to view insights:
   ```
   market Dash board.pbix
   ```
4. Check SQL queries in the `SQL_queries` folder for data exploration.

## Project Structure
```
Stock-Analysis/
├── data/               # Raw stock data files
├── SQL_queries/        # SQL query files for data analysis
├── main.py             # Main Python script for analysis
├── market Dash board.pbix # Power BI dashboard file
├── yaml_to_csv.ipynb   # Jupyter notebook for data cleaning
└── README.md           # Project documentation
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License
This project is licensed under the MIT License.

## Acknowledgements
Special thanks to all the open-source contributors and the community for their support.

