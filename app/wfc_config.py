import json

from logzero import logger


class wfc_pfolio:

    def __init__(self):
        self.name = "None"

    def fromjson(self, cfg):
        self.name    = cfg['NAME']
        self.tkrs    = cfg['TICKERS']   # reference into cfg_data; mutations persist on save()
        self.related = cfg['RELATED']
        self.exchg   = cfg['EXCHG']

    def tickers(self):
        return self.tkrs

    def add_ticker(self, ticker):
        if ticker not in self.tkrs:
            self.tkrs.append(ticker)

    def del_ticker(self, ticker):
        if ticker in self.tkrs:
            self.tkrs.remove(ticker)


class wfc_config:

    def __init__(self, cfg_file):
        self.cfg_data  = json.loads(cfg_file)
        self.pf_dict   = {}
        self.user      = self.cfg_data.get('basename', '')
        self.password  = self.cfg_data.get('password', '')
        self.data_loc  = self.cfg_data.get('data_location', 'data/')
        self.purchased = self.cfg_data.get('purchased', [])

        for x in self.cfg_data.get('pfolios', []):
            self.pf_dict[x['NAME']] = wfc_pfolio()
            self.pf_dict[x['NAME']].fromjson(x)

    def pf(self, pf_key):
        return self.pf_dict[pf_key]

    def show(self):
        for name, pf in self.pf_dict.items():
            print(f"{name}\t{pf.exchg}\t{pf.tickers()}")

    def data_location(self):
        return self.data_loc

    def save(self, cfg_file):
        with open(cfg_file, 'w') as f:
            json.dump(self.cfg_data, f, indent=4, sort_keys=True)

    # ------------------------------------------------------- portfolio management

    def portfolio_add(self, pf_name, exchg="NSDQ"):
        if pf_name in self.pf_dict:
            return False
        entry = {"NAME": pf_name, "TICKERS": [], "RELATED": [], "EXCHG": exchg}
        self.cfg_data.setdefault('pfolios', []).append(entry)
        self.pf_dict[pf_name] = wfc_pfolio()
        self.pf_dict[pf_name].fromjson(entry)
        return True

    def portfolio_del(self, pf_name):
        if pf_name not in self.pf_dict:
            return False
        self.cfg_data['pfolios'] = [
            p for p in self.cfg_data.get('pfolios', []) if p['NAME'] != pf_name
        ]
        del self.pf_dict[pf_name]
        return True

    # --------------------------------------------------------- ticker management

    def ticker_add(self, pf_name, ticker):
        if pf_name in self.pf_dict:
            self.pf_dict[pf_name].add_ticker(ticker)
        else:
            logger.error(f"Portfolio not found: {pf_name}")

    def ticker_del(self, pf_name, ticker):
        if pf_name in self.pf_dict:
            self.pf_dict[pf_name].del_ticker(ticker)
        else:
            logger.error(f"Portfolio not found: {pf_name}")

    # ---------------------------------------------------------- owned management

    def owned(self, args):
        if not args:
            self.owned_list()
            return
        cmd = args[0]
        if cmd == "add" and len(args) >= 5:
            self.owned_add(args[1], args[2], args[3], args[4])
        elif cmd == "del" and len(args) >= 3:
            self.owned_del(args[1], args[2])
        else:
            print("owned [ add | del ] <ticker> <purchase_date> <share_count> <price_paid>")

    def owned_list(self):
        for x in self.purchased:
            print(x['ticker'])
            for i in x['ticker_purchase']:
                for date, details in i.items():
                    cost = int(details[0] * details[1])
                    print(f"\t{date}\t${cost:,}\t{details}")

    def isOwned(self, ticker):
        return any(x['ticker'] == ticker for x in self.purchased)

    def owned_ticker(self, ticker):
        for x in self.purchased:
            if x['ticker'] == ticker:
                return x
        return None

    def owned_add(self, ticker, purchase_date, share_count, buy_price):
        entry = {purchase_date: [int(share_count), float(buy_price)]}
        if self.isOwned(ticker):
            for x in self.purchased:
                if x['ticker'] == ticker:
                    x['ticker_purchase'].append(entry)
                    return
        else:
            self.purchased.append({
                'ticker': ticker,
                'ticker_purchase': [entry],
            })

    def owned_del(self, ticker, purchase_date):
        if not self.isOwned(ticker):
            return
        for x in self.purchased:
            if x['ticker'] == ticker:
                x['ticker_purchase'] = [
                    i for i in x['ticker_purchase']
                    if purchase_date not in i
                ]
                if not x['ticker_purchase']:
                    self.purchased.remove(x)
                return
