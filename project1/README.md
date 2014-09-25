# Benchmark Bond Trade Price Challenge
(Kaggle competition: https://www.kaggle.com/c/benchmark-bond-trade-price-challenge)

===
### Goal
The Benchmark Bond Trade Price Challenge is a competition to predict the next price that a US corporate bond might trade at. Contestants are given information on the bond including current coupon, time to maturity and a reference price computed by Benchmark Solutions.  Details of the previous 10 trades are also provided.  

=====
### Background

As far as price transparency is concerned, there has historically been a huge gap between the amount of reference information available to those trading equities versus those trading corporate bonds.  Stock exchanges report trades, bids and offers at all times.  Free access is available online with a 15 minute delay while traders who demand more information can pay for ultra efficient real time data and information about size of current bids and offers. By contrast, bond trades are required to be reported within 15 minutes and only those who pay for the TRACE feed can access this information.  No quotes are publicly available and the best way to get a quote is to solicit multiple brokers and wait for a reply.  Alternatively there are data companies that provide end of day prices, published after the market has closed and with no guarantee that the specific information sought will be included.  Accurate bond pricing is also hindered by lack of liquidity.  Only a fraction of TRACE eligible bonds trade on a given day, so the most recent trade price is often multiple days old.  Pricing bonds based on other more liquid bonds that have similar features is common, but again limited by the presence of such bonds.

Benchmark Solutions is the first provider of realtime corporate bond prices. Every 10 seconds we provide accurate prices that incorporate interest rate data, trades or quotes of the bond in question, trades or quotes of other bonds or CDS of the issuer of the bond in question as well as other input sources.  Pricing bonds accurately requires an exacting knowledge of payment schedules, trading calenders and reference data for each bond.  This, as well as synthesizing all of the bonds and CDS quotes and trades of a given issuer into implied hazard and funding curves, is something that we feel is beyond the scope of this challenge.  Rather, we provide you with a reference price which is an intermediate result of our calculations and is labeled 'curve_based_price'  in the dataset. Thus the competition focuses on trading dynamics and microstructure of individual bonds, rather than all bonds from a given issuer.  

===
### Variables
US corporate bond trade data is provided.  Each row includes trade details, some basic information about the traded bond, and information about the previous 10 trades.  Contestants are asked to predict trade price.

Column details:

- id: The row id.
- bond_id: The unique id of a bond to aid in time-series reconstruction (this column is only present in the train data)
- trade_price: The price at which the trade occured.  (this is the response variable, i.e. the column to predict in the test data)
- weight: The weight of the row for evaluation purposes. This is calculated as the square root of the time since the last trade and then scaled so the mean is 1.
- current_coupon: The coupon of the bond at the time of the trade.
- time_to_maturity: The number of years until the bond matures at the time of the trade.
- is_callable: A binary value indicating whether or not the bond is callable by the issuer (categorical variable).
- reporting_delay: The number of seconds after the trade occured that it was reported.
- trade_size: The notional amount of the trade.
- trade_type: 2=customer sell, 3=customer buy, 4=trade between dealers. We would expect customers to get worse prices on average than dealers (categorical variable).
- curve_based_price: A fair price estimate based on implied hazard and funding curves of the issuer of the bond.
- received_time_diff_last{1-10}: The time difference between the trade and that of the previous {1-10}.
- trade_price_last{1-10}: The trade price of the last {1-10} trades.
- trade_size_last{1-10}: The notional amount of the last {1-10} trades.
- trade_type_last{1-10}: The trade type of the last {1-10} trades (categorical variable).
- curve_based_price_last{1-10}: The curve based price of the last {1-10} trades.
