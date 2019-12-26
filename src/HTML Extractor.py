from bs4 import BeautifulSoup as bs
import re

page = open('/Users/Ju1y/Documents/GIES Research Project/10-K/2010Q1/1800-ABBOTT LABORATORIES-10-K-2010-02-19.html')
soup = bs(page.read(), 'html.parser')

locators = soup.findAll(text=re.compile(r'FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY\s*DATA', re.IGNORECASE))
item_8_start = ''
tag_name = ''
item_8_found = False

if locators:
    for data in locators:
        if data.parent.name == 'b':
            tag_name = 'b'
            item_8_found = True
        elif data.parent.name == 'font' and data.parent.has_attr('style') and 'FONT-WEIGHT: bold' in data.parent['style']:
            tag_name = 'font'
            item_8_found = True
        if item_8_found:
            while data.parent.name != 'body':
                data = data.parent
            item_8_start = data
            print('tag_name: ', tag_name)
            print(item_8_start)
            break;
else:
    print('No keyword found, check document manually')


if item_8_found:
    if tag_name == 'b':

