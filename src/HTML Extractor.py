from bs4 import BeautifulSoup as bs
import re

page = open('/Users/Ju1y/PycharmProjects/10-K-Form-Analyzer/Test_HTML/1800-ABBOTT LABORATORIES-10-K-2010-02-19.html')
soup = bs(page.read(), 'html.parser')

trial1 = soup.findAll(text=re.compile('FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA'))
trial2 = soup.select_one('font[FONT-WEIGHT~="BOLD"]', text=re.compile('FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA'))

if trial1:
    for data in trial1:
        print(data)
        print(data.parent)
        print(data.parent.name == 'b')
elif trial2:
    print('Trial 2 found')
    print(trial2)
else:
    print('No Trial found, check document manually')


