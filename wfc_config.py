import json

from logzero import logger
from logzero import logging

class wfc_pfolio:

    def __init__(self):
        self.name = "None"

    def fromjson(self, cfg):
        self.name = cfg['NAME']
        self.tkrs = cfg['TICKERS']
        self.related = cfg['RELATED']
        self.exchg = cfg['EXCHG']

    def tickers(self):
        return self.tkrs

class wfc_config:

    def __init__(self, cfg_file):
        self.cfg_data = json.loads(cfg_file)
        self.pf_dict = {}
        self.user = self.cfg_data['basename']
        self.password = self.cfg_data['password']
        self.data_loc = self.cfg_data['data_location']
        self.purchased = self.cfg_data['purchased']

        pfolios = self.cfg_data['pfolios']
        for x in pfolios:
            self.pf_dict[ x['NAME'] ]  = wfc_pfolio()
            self.pf_dict[ x['NAME'] ].fromjson(x)

    def pf(self, pf_key):
        return self.pf_dict[pf_key]

    def show(self):
        for x in self.pf_dict:
            el = self.pf_dict[x]
            print(x + "\t" + str(el.exchg) + "\t" + str(el.tkrs))

    def data_location(self):
        return self.data_loc

    def owned(self, args):
        if (len(args) == 0):
            self.owned_list()
        else:
            cmd = args[0]
            if (cmd == "add"):
                ticker = args[1]
                pdate = args[2]
                stock_count = args[3]
                pprice = args[4]
                self.owned_add(ticker, pdate, stock_count, pprice)
            elif (cmd == "del"):
                ticker = args[1]
                pdate = args[2]
                self.owned_del(ticker, pdate)
            else:
                print("owned [ add | del ] <ticker> <purchase_date> [ <share_count> <price_paid> ]")

    def owned_list(self):
        tlist = self.purchased
        for x in tlist:
            print(str(x['ticker']))
            for i in x['ticker_purchase']:
                k = i.keys()
                for kk in k:
                    pd = i[kk]
                    pr = int(pd[0] * pd[1])
                    print("\t" + str(kk) + "\t" + str(pr) + "\t" + str(pd))

    def isOwned(self, ticker):
        tlist = self.purchased
        for x in tlist:
            if ( x['ticker'] == ticker ):
                return True
        return False

    def save(self, cfgFile):
        with open(cfgFile, 'w') as outfile:
            json.dump(self.cfg_data, outfile, indent=4, sort_keys=True)

    def owned_ticker(self, ticker):
        if (self.isOwned(ticker)):
            tlist = self.purchased
            for x in tlist:
                if ( x['ticker'] == ticker ):
                    return x

    def owned_add(self, ticker, purchase_date, share_count, buy_price):
        print(ticker + "\t" + purchase_date + "\t" + share_count + "\t" + buy_price)
        if (self.isOwned(ticker)):
            tlist = self.purchased
            for x in tlist:
                if ( x['ticker'] == ticker ):
                    dt_item = {}
                    pd_item = []
                    pd_item.append(int(share_count))
                    pd_item.append(float(buy_price))
                    dt_item[purchase_date] = pd_item
                    x['ticker_purchase'].append(dt_item)
                    return
        else:
            tlist = self.purchased
            lastnum = len(tlist)
            item = {}
            item['ticker'] = ticker
            item['ticker_purchase'] = []
            dt_item = {}
            pd_item = []
            pd_item.append(int(share_count))
            pd_item.append(float(buy_price))
            dt_item[purchase_date] = pd_item
            item['ticker_purchase'].append(dt_item)
            self.purchased.append(item)

    def owned_del(self, ticker, purchase_date):
        print(ticker + "\t" + purchase_date)
        if (not self.isOwned(ticker)):
            return
        tlist = self.purchased
        for x in tlist:
            if ( x['ticker'] == ticker ):
                for i in x['ticker_purchase']:
                    k = i.keys()
                    for kk in k:
                        if (kk == purchase_date):
                            del i[kk]
                            x['ticker_purchase'].remove(i)
                            if (len(x['ticker_purchase']) == 0):
                                self.purchased.remove(x)
                            return

