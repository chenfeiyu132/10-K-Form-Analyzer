# 10-K-Form-Analyzer
Python code designed to extract sections from company 10-K forms and apply machine learning classifiers to identify certain legal language mentioned in text

## Purpose: 
The purpose of this project is to analyze past 10-K forms and certain legal information located within them using a machine learning/data science approach. 10-K forms are annual reports required to be filed by companies to its shareholders. These reports are often lengthy but often contain snippets of useful information that could be used to provide insights into some of the companies’ financial as well as legal situations. Thus, these useful little “gold nuggets” inside 10-K’s often could potentially serve as early indicators to the market directions in the companies’ shares before certain information like prosecution results make the headlines. 

One difficulty with the analysis of these documents, which is also one of the bigger challenges to Natural Language Processing, is the presence of ambiguous legal languages which are often employed by companies to mitigate the severity of the legal problems they are facing. Thus, one primary target of this project is to process and filter out important indicative terms from 10-K text that are often associated with prosecution. 

## Data Collection:
The data being used are 10-K forms scraped from the SEC EDGAR database from 2019 to 2007. Due to the sheer sizes of the 10-K’s, the key focus of the data analysis will be only on section Item 8, “Financial Statements and Supplementary Data,” since researchers have found that companies often disclose legal information in this section.
  
## Procedure:
The ultimate goal of the project is not only to isolate important terms associated with prosecution but also train machine learning models that could be used to identify 10-K documents in the future in terms of legal situations.

## Pipeline Flowchart:
<p align="center">
  ![Image of Flowchart](https://github.com/chenfeiyu132/10-K-Form-Analyzer/blob/master/10-K%20diagram.png)
</p>
