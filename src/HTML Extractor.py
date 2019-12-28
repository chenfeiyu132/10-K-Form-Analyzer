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


def find_root_parent(element):  # finds the top most parent for certain element under body
    while element.parent and element.parent.name != 'text':
        element = element.parent
        #print(element.name)
    return element


def locate_item(regex):

    locators = soup.findAll(text=re.compile(regex, re.IGNORECASE))
    if locators:
        for tag in locators:
            if tag.parent.name == 'b':
                print('tag_name: ', 'b')  # debugging purposes
                return find_root_parent(tag)
            elif tag.parent.name == 'font' and tag.parent.has_attr('style') \
                    and 'FONT-WEIGHT: bold' in tag.parent['style']:
                        print('tag_name: ', 'font')
                        return find_root_parent(tag)
    print('item not found')
    return None


def locate_next_item(tag, regex):
    locators = tag.find_all_next(text=re.compile(regex, re.IGNORECASE))
    if locators:
        for t in locators:
            if tag.find('b') and t.parent.name == 'b':
                return find_root_parent(t)
            elif t.parent.name == 'font' and t.parent.has_attr('style') and 'FONT-WEIGHT: bold' in t.parent['style']:
                return find_root_parent(t)
    print('next item not found')
    return None


item_8_start = locate_item(r'FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY\s*DATA')
item_9_start = locate_next_item(item_8_start, r'CHANGES\s*IN\s*AND\s*DISAGREEMENTS\s*WITH\s*'
                                r'ACCOUNTANTS\s*ON\s*ACCOUNTING\s*AND\s*FINANCIAL\s*'
                                r'DISCLOSURE')

delete_section('', item_8_start)
delete_section(item_9_start, '')
item_9_start.extract()

print(item_8_start.previous_sibling.previous_sibling)  # delete privacy message

# for doc in soup.body.findAll('document'):
#     doc.extract()

with open('test_output_1.html', 'w', encoding='utf-8') as file:
    file.write(str(soup))



