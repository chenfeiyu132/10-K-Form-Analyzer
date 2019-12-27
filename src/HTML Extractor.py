from bs4 import BeautifulSoup as bs
import re

# Process outline:
    # locate the start of item_8
    # locate the start of item_9
    # extract/delete anything before start of item_8 and after start of item_9
    # output new html after extraction

page = open('/Users/Ju1y/Documents/GIES Research Project/10-K/2010Q1/1800-ABBOTT LABORATORIES-10-K-2010-02-19.html')
soup = bs(page.read(), "lxml")

item_8_start = ''
item_9_start = ''
tag_name = ''


def delete_section(beginning, end):  # deletes section based on beginning html tag and end html tag
    if beginning == '':
        beginning = soup.body.contents[0]
    if end == '':
        end = soup.body.contents[-1]
    for elm in beginning.find_next_siblings:
        if elm != end:
            elm.extract()
        else:
            break


def find_root_parent(element):  # finds the top most parent for certain element under body
    while element.parent and element.parent.name != 'text':
        element = element.parent
        print(element.name)
    return element


item_8_locators = soup.findAll(text=re.compile(r'FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY\s*DATA', re.IGNORECASE))
if item_8_locators:
    for data in item_8_locators:
        if data.parent.name == 'b':  # case is <b></b> is used for the bold font
            tag_name = 'b'
            item_8_start = find_root_parent(data)
            print('tag_name: ', tag_name)  # debugging purposes
            #print(item_8_start)  # debugging purposes
            item_9_locators = item_8_start.find_all_next(text=re.compile(r'CHANGES\s*IN\s*AND\s*DISAGREEMENTS\s*WITH\s*ACCOUNTANTS\s*ON\s*ACCOUNTING\s*AND\s*FINANCIAL\s*DISCLOSURE', re.IGNORECASE))
            for locator in item_9_locators:
                if locator.parent.name == 'b':
                    item_9_start = find_root_parent(locator)
                    break
            break
        elif data.parent.name == 'font' \
                and data.parent.has_attr('style') \
                and 'FONT-WEIGHT: bold' in data.parent['style']:  # case if <font></font> is used for bold font
            tag_name = 'font'
            item_8_start = find_root_parent(data)
            print('tag_name: ', tag_name)  # debugging purposes
            #print(item_8_start)  # debugging purposes
            item_9_locators = item_8_start.find_all_next(text=re.compile(r'CHANGES\s*IN\s*AND\s*DISAGREEMENTS\s*WITH\s*'
                                                                 r'ACCOUNTANTS\s*ON\s*ACCOUNTING\s*AND\s*FINANCIAL\s*'
                                                                 r'DISCLOSURE',
                                                                 re.IGNORECASE))
            for locator in item_9_locators:
                if locator.parent.name == 'font' and locator.parent.has_attr('style') and 'FONT-WEIGHT: bold' in locator.parent['style']:
                    item_9_start = find_root_parent(locator)
                    break
            break
else:
    print('No keyword found, check document manually')




