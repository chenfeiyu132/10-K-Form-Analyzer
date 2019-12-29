from bs4 import BeautifulSoup as bs
import re

# Process outline:
    # locate the start of item_8
    # locate the start of item_9
    # extract/delete anything before start of item_8 and after start of item_9
    # output new html after extraction

page = open('/Users/Ju1y/Documents/GIES Research Project/10-K/2010Q1/2098-ACME UNITED CORP-10-K-2010-03-09.html')
soup = bs(page.read(), "lxml")

item_8_start = ''
item_9_start = ''


def delete_section(beginning, end):  # deletes section based on beginning html tag and end html tag
    if beginning == '' and end == '':
        return
    elif beginning == '':
        for elm in end.find_previous_siblings():
            elm.extract()
    elif end == '':
        for elm in beginning.find_next_siblings():
            elm.extract()
    else:
        for elm in beginning.find_next_siblings():
            if elm != end:
                elm.extract()
            else:
                break


def find_root_parent(element, grandparent):  # finds the top most parent for certain element under body
    while element.parent and element.parent.name != grandparent:
        element = element.parent
    return element


def locate_item(regex, grandparent):

    locators = soup.findAll(text=re.compile(regex, re.IGNORECASE))
    if locators:
        for tag in locators:
            if tag.parent.name == 'b':
                print('tag_name: ', 'b')  # debugging purposes
                return find_root_parent(tag, grandparent)
            elif tag.parent.name == 'font' and tag.parent.has_attr('style') \
                    and 'FONT-WEIGHT: bold' in tag.parent['style']:
                        print('tag_name: ', 'font')  # debuging purposes
                        return find_root_parent(tag, grandparent)
    print('item not found')
    return None


def locate_next_item(tag, regex, grandparent):
    locators = tag.find_all_next(text=re.compile(regex, re.IGNORECASE))
    if locators:
        for t in locators:
            if tag.find('b') and t.parent.name == 'b':
                return find_root_parent(t, grandparent)
            elif t.parent.name == 'font' and t.parent.has_attr('style') and 'FONT-WEIGHT: bold' in t.parent['style']:
                return find_root_parent(t, grandparent)
    print('next item not found')
    return None


item_8_start = locate_item(r'FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY\s*DATA', 'text')
item_9_start = locate_next_item(item_8_start, r'CHANGES\s*IN\s*AND\s*DISAGREEMENTS\s*WITH\s*'
                                r'ACCOUNTANTS\s*ON\s*ACCOUNTING\s*AND\s*FINANCIAL\s*'
                                r'DISCLOSURE', 'text')


delete_section('', item_8_start)  # deletes everything up to item 8
delete_section(item_9_start, '')  # deletes everything after start of item 9
item_9_start.extract()  # deletes start of item 9

item_document = locate_item(r'FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY\s*DATA', 'sec-document')

delete_section(item_document, '')  # deletes every other document besides the one which item 8 is located in

with open('test_output_2.html', 'w', encoding='utf-8') as file:  # output processed soup into html
    file.write(str(soup))



