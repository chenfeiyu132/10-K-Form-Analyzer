from bs4 import BeautifulSoup as bs
import re
import os
import copy

# Process outline:
    # locate the start of item_8
    # locate the start of item_9
    # extract/delete anything before start of item_8 and after start of item_9
    # output new html after extraction


def find_root_parent(tag_1, reference):  # finds the top most parent for certain element under body
    if not tag_1:
        return tag_1
    while tag_1.parent and reference not in tag_1.parent.descendants:
        tag_1 = tag_1.parent
    return tag_1


def locate_item(regex, soup):

    locators = soup.findAll(text=re.compile(regex, re.IGNORECASE))
    if locators:
        for tag in locators:
            bold = tag.find_parent('b')
            font = tag.find_parent('font', style=re.compile(r'FONT-WEIGHT: bold'))
            if bold:
                print('tag_name: ', 'b')  # debugging purposes
                return bold
            elif font:
                print('tag_name: ', 'font')  # debuging purposes
                return font
    print('item not found')
    return None


def append_tags(begin, end, soup):
    if not begin:
        return soup
    for tag in begin.find_next_siblings():
        if tag != end:
            soup.append(tag)
        else:
            break
    return soup


def process_html(path, output_folder_path):
    page = open(path)
    soup = bs(page.read(), "lxml")
    file_name = os.path.basename(path)

    item_8_start = locate_item(r'FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY\s*DATA', soup)
    item_9_start = locate_item(r'(ITEM)?.*?9?.*?(CHANGES)?.?\s*(IN)?.?\s*(AND)?.?\s*(DISAGREEMENTS)?.?\s*(WITH)?.?\s*'
                                    r'(ACCOUNTANTS)?.?\s*(ON)?.?\s*(ACCOUNTING)?.?\s*(AND)?.?\s*(FINANCIAL)?.?\s*'
                                    r'(DISCLOSURE)?.?', soup)
    item_8_parent = find_root_parent(item_8_start, item_9_start)
    item_9_parent = find_root_parent(item_9_start, item_8_start)
    new_soup = bs(features='lxml')
    if item_8_start and item_9_start:

        new_soup = append_tags(item_8_parent.previous_sibling, item_9_parent, new_soup)
        output_path = output_folder_path + file_name
        #print(output_path)
        with open(output_path, 'w', encoding='utf-8') as file:  # output processed soup into html
            file.write(str(new_soup))


directory = '/Users/Ju1y/Documents/GIES Research Project/10-K/2010Q1/'
output_folder = '/Users/Ju1y/Documents/GIES Research Project/Item8/'

count = 0


for filename in os.listdir(directory):
    if filename.endswith('.html'):
        print(filename)
        process_html(directory+filename, output_folder)
    elif filename.endswith('.txt') and count < 20:
        print(filename)
        pre, ext = os.path.splitext(directory+filename)
        os.rename(directory+filename, pre + '.html')
        process_html(pre + '.html', output_folder)
        count += 1
