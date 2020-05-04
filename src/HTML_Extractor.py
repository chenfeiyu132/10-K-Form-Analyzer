from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
import re
import os


def find_root_parent(tag_1, reference):  # finds the top most parent for certain element under body
    if not tag_1:
        return tag_1
    while tag_1.parent and reference not in tag_1.parent.descendants:
        tag_1.parent.unwrap()
    return tag_1


def locate_item(regex, soup):

    locators = soup.findAll(text=re.compile(regex, re.IGNORECASE))
    if locators:
        for tag in locators:
            bold = tag.find_parent('b')
            font = tag.find_parent(style=re.compile(r'FONT-WEIGHT:\s*bold', re.IGNORECASE))
            if bold:
                print('tag_name: ', 'b')  # debugging purposes
                return bold
            elif font:
                print('tag_name: ', 'font')  # debuging purposes
                return font
    print('item not found')
    return None


def between(cur, end):
    while cur and cur != end:
        if isinstance(cur, NavigableString):
            text = cur.strip()
            if len(text):
                yield text
        cur = cur.next_element


def backup_locator(item8_regex, item9_regex, soup):
    print("initiating backup plan")
    locators = soup.findAll(text=re.compile(item8_regex, re.IGNORECASE))
    if locators:

        max_length = 0
        correct_start = ''
        correct_end = ''

        for start_title in locators:
            end_title = start_title.findNext(text=re.compile(item9_regex, re.IGNORECASE))
            if not end_title:
                break;
            # very important that you use the parent of the located titles
            start_parent = find_root_parent(start_title.parent, end_title.parent)
            end_parent = find_root_parent(end_title.parent, start_title.parent)
            btw = ' '.join(text for text in between(start_parent, end_parent))
            if len(btw) > max_length:
                max_length = len(btw)
                correct_start = start_parent
                correct_end = end_parent

        if max_length != 0:
            #print(correct_start.name)
            #print(correct_end.name)
            return append_tags(correct_start.previous_sibling, correct_end, bs(features='lxml'))

    print("backup failed, nothing was found/located")


def append_tags(begin, end, soup):
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
    item_8_start = locate_item(r'((ITEM)\s*8)|FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY?\s*DATA', soup)
    item_9_start = locate_item(r'((ITEM)\s*9)|(CHANGES).?\s*(IN).?\s*(AND).?\s*(DISAGREEMENTS).?\s*(WITH).?\s*'
                                    r'(ACCOUNTANTS).?\s*(ON).?\s*(ACCOUNTING).?\s*(AND).?\s*(FINANCIAL).?\s*'
                                    r'(DISCLOSURE).?', soup)
    new_soup = bs(features='lxml')
    if item_8_start and item_9_start:
        item_8_parent = find_root_parent(item_8_start, item_9_start)
        item_9_parent = find_root_parent(item_9_start, item_8_start)
        new_soup = append_tags(item_8_parent.previous_sibling, item_9_parent, new_soup)
    else:
        new_soup = backup_locator(r'((ITEM)\s*8)|FINANCIAL\s*STATEMENTS\s*AND\s*SUPPLEMENTARY?\s*DATA',
                       r'((ITEM)\s*9)|(CHANGES).?\s*(IN).?\s*(AND).?\s*(DISAGREEMENTS).?\s*(WITH).?\s*'
                       r'(ACCOUNTANTS).?\s*(ON).?\s*(ACCOUNTING).?\s*(AND).?\s*(FINANCIAL).?\s*'
                       r'(DISCLOSURE).?',
                       soup)
    if not new_soup:
        return

    output_path = output_folder_path + file_name
    with open(output_path, 'w', encoding='utf-8') as file:  # output processed soup into html
        file.write(str(new_soup))


def convert_html(directory_name, output_folder_name):
    for filename in os.listdir(directory_name):
        if filename.endswith('.html'):
            print(filename)
            process_html(directory_name+filename, output_folder_name)
        elif filename.endswith('.txt'):
            print(filename)
            pre, ext = os.path.splitext(directory_name+filename)
            os.rename(directory_name+filename, pre + '.html')
            process_html(pre + '.html', output_folder_name)



