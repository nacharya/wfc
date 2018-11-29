
wforecast: ( Wealth Forecast ) - a Stock prediction application
---------------------------------------------------------------
Allows you to create a portfolio, add delete buy/sell info
Pulls data from Yahoo Finance. 
Makes predictions based on the data collected 
Makes Financial assesment based on what is in the portfolio

- Create a base config from the sample here in the repo

	% cp dot_wfc.sample ~/.wfc

- Run it commandline e.g.

	% wfc 
	% wfc --help
	% wfc --system
	% wfc list
	% wfc gather nasdaq 10-30-2018 12-10-2018 
	% wfc -d 0 predict NFLX 12-21-2018
	% wfc news NFLX 
	% wfc show NFLX now
	% wfc show owned
	% wfc owned add AAPL 9-15-2018 100 217 

- or Run a Jupyter notebook
	% jupyter-lab

	Open wfc_doc.ipynb

