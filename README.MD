# What is Composer.trade?

Composer.trade is a no-code trading platform that enables users to create, automate, and execute custom investment strategies. It simplifies algorithmic trading by providing an intuitive drag-and-drop interface, pre-built templates, and powerful backtesting capabilities. This allows users with minimal technical expertise to access advanced trading strategies previously reserved for professional traders and developers.

## Key Features:
- **No-code interface**: Drag-and-drop strategy builder to design complex trading algorithms.
- **Backtesting**: Evaluate the performance of strategies using historical data.
- **Pre-built templates**: Start with proven strategies and customize them.
- **Automation**: Deploy strategies that execute trades automatically based on set rules.
- **Integration with brokers**: Seamlessly connect with supported brokerage accounts for execution.

---

## Cons of Composer.trade vs Competitors

1. **Limited Broker Integrations**:
   - The actual trading platform (Alpaca) is hidden from the user.

2. **Feature Depth**:
   - Some advanced features like direct API integrations or deep customization are less robust compared to platforms like QuantConnect.

3. **Scalability**:
   - May not cater well to professional traders requiring **high-frequency** trading or institutional-grade tools. Only trade ONCE per day.

4. **Cost for Advanced Users**:
   - Pricing could become less competitive for users needing extensive customizations or additional features.

5. **Learning Curve**:
   - While simpler than coding, users new to trading might still need time to understand strategy development.

6. **Dependence on Platform**:
   - Strategies are platform-dependent, limiting portability to other services.

---

## Comparison Table

| Feature               | Composer.trade            | QuantConnect                | Tradestation              |
|-----------------------|---------------------------|-----------------------------|---------------------------|
| No-code Interface     | Yes                       | No                          | Partial                   |
| Backtesting           | Yes (User-friendly)       | Yes (Advanced)              | Yes (Intermediate)        |
| Pre-built Strategies  | Yes                       | No                          | Yes                       |
| Broker Integrations   | Limited                   | Broad                       | Broad                     |
| Cost                  | Subscription-based        | Free for basic use          | Commission-based          |
| Advanced Features     | Limited                   | Extensive                   | Moderate                  |

---

## Scripts
1. composer_db.ipynb - the script pulls algo data from the public composer.trade API and store them into CSV files
2. composer_db_analysis - the script contains an example of how to filter out the algos from the CSV files based on specific requirements
3. composer_db_analysis_outdated.ipynb - the script contains an example of how to filter out the algos from the CSV files based on specific requirements

## CSV db examples


## Tools
- [API Doc](https://backtest-api.composer.trade/api/v2/api-docs/index.html)
- [Algo Analyzer](https://mymaestro.co/)
- [ETF database](https://www.composer.trade/etf)
