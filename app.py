import nltk
# nltk.data.path.append('C:\\Users\\Divdawar\\AppData\\Roaming\\nltk_data')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, render_template, request, jsonify
import pandas as pd
# from GoogleNews import GoogleNews
# from newsapi import NewsApiClient
from gnews import GNews
from newspaper import Article
from newspaper.article import ArticleException
from rake_nltk import Rake
import re
import time
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from newspaper import Config
import tempfile
from newspaper import Article
import datetime
# from summarizer import Summarizer
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import io
import base64
# import requests

# requests.get('https://example.com', verify=True, tls_version='TLSv1.2')


app = Flask(__name__)


# newsapi = NewsApiClient(api_key='6ff27af7a3ab4d33a995573ea4b15d8c')

# Initialize the LSA summarizer
summarizer = LsaSummarizer()
# Number of sentences in the summary
num_sentences = 4
num_sentences1= 8
# Introduce a delay of 2 seconds between each API call
delay_between_requests = 2

# keywords = ["order", "worth", "power", "solar", "mw", "project", "power transmission", "billion", "bags", "green energy", "power commission", "green hydrogen", "secures order", "energy", "worth crore", "bags", "transmission", "distribution", "substation", "renewable", "commission", "awarded", "bidding", "tender", "received", "Power grid", "Transmission lines", "Distribution network", "Renewable energy", "Solar power", "Wind power", "Hydroelectric power", "Geothermal energy", "Biomass energy", "Energy storage", "Grid integration", "Clean energy", "Net metering", "Feed-in tariffs", "Power sector reforms", "Electric utilities", "Energy transition", "Energy policy", "Carbon footprint", "Green technology"]

keywords= ["Power", "Solar", "Energy", "Tender", "Offshore wind", "Wins", "Green Energy", "Commissioned", "Energy Policy"]

prefixed_keywords = ["Power", "Transmission", "Solar", "Renewable", "Energy", "Project", "Wins", "Offshore Wind", "Order","Secures","Construction","EPC"]


# prefixed_keywords = ["business","power","transmission","solar","renewable","energy","project","wins","offshore wind","order"]

# competitors = ["KPTL", "Adani", "Skipper", "Avaada", "TATA", "KEC", "Tata Power", "Suzlon", "Jakson Green", "SunEdison", "Renew", "Inox Green", "Sun Source Energy", "Greenko", "Zetwerk", "Infrastructure", "Kalpatru", "Sterling", "IndiGrid", "NCLT", "Jakson", "Kalpataru", "Siemens", "Sterlite", "GE", "Infosys", "Bajaj", "Hartek", "Waaree", "Roofsol", "BVG", "Oriana", "Power", "Linxon"]
competitors=['Adani','Avaada Energy','Hartek','Kalpataru Power','Tata Power','KEC','Sterlite','Zetwerk','ReNew','IndiGrid']
competitor_image=['adani.jpg','avaada.jpg','hartek.jpg','Kalpataru Power.png','Tata Power.jpg','KEC International.png','Sterlite.jpg','Zetwerk.jpg','ReNew Power.png','IndiGrid.jpg']
# customers= ['Power Grid', 'Gencos', 'Transcos', 'Discoms', 'MOP', 'MNRE', 'REC', 'PFC', 'ADB', 'JICA', 'WB', 'MOSPI', 'CEA', 'Niti Aayog', 'Draft Electricity plan', 'NSE', 'BSE']
customers= ['ACWA','Power Grid','Gencos','Discom','Ministry of Power','Ministry of New and Renewable Energy','REC','PFC','Asian Development Bank','Central Electricity Authority']
customer_image = ['ACWA.png','Power Grid.jpg','Gencos.png','Discom.png','Ministry of Power.jpg','Ministry of New and Renewable Energy.jpg','REC.png','PFC.png','ADB.png','Central Electricity Authority.png']

manufacturers = ['Jinko','Sungrow','BYD','Nextracker']
manufacturers_image = ['jinko.png','sungrow.png','BYD.png','nextracker.png']

# proxy = 'http://proxy.example.com:8080'

# Function to check sentence similarity between two news articles
def check_sentence_similarity(article1, article2, similarity_threshold):
    
    article1 = article1.lower()
    article2 = article2.lower()

    sentences1 = sent_tokenize(article1)
    sentences2 = sent_tokenize(article2)

    similarity_count=0

    for sentence1 in sentences1:
        for sentence2 in sentences2:
            similarity_score = SequenceMatcher(None, sentence1, sentence2).ratio()
            
            if similarity_score >= similarity_threshold:
                print('Similarity in sentences :',similarity_score)
                similarity_count += 1
    
    if similarity_count>len(sentences1)/2:
        return True

    return False

# Function to check word similarity between two news articles
def check_word_similarity(article1, article2, similarity_threshold):

    article1 = article1.lower()
    article2 = article2.lower()

    words1 = set(word_tokenize(article1))
    words2 = set(word_tokenize(article2))

    lemmatizer = WordNetLemmatizer()
    lemmatized_words1 = set(lemmatizer.lemmatize(word) for word in words1)
    lemmatized_words2 = set(lemmatizer.lemmatize(word) for word in words2)

    common_words = lemmatized_words1.intersection(lemmatized_words2)
    if len(words1)>0:
        similarity = len(common_words) / len(words1)
        
    else:
        similarity=0

    if similarity >= similarity_threshold:
        print('Similarity in words :',similarity)
        return True

    return False


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
# Function to lemmatize a sentence
def lemmatize_sentence(sentence):
    words = word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

@app.route('/')
def index():
    return render_template('index.html', keywords=keywords, customers=customers, competitors=competitors,competitor_image=competitor_image,customer_image=customer_image,manufacturers=manufacturers,manufacturers_image=manufacturers_image)



# from flask import Flask, render_template, jsonify
# import requests

@app.route('/fetch_stock_data')
def fetch_stock_data():
    url = 'https://www.equitymaster.com/stockquotes/Power/list-of-power-sector'
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='tblSectorData')

        if table:
            rows = table.find_all('tr')[1:]
            # Initialize lists to store data
            data = []

            # Loop through each row
            for row in rows:
                # Find all table cells in the row
                cells = row.find_all(['td', 'th'])
                # Extract text from each cell and strip whitespace
                row_data = [cell.get_text(strip=True) for cell in cells]
                # Add symbols for BSE and NSE price columns
                if len(row_data) >= 3:  # Ensure there are at least 3 columns (BSE, NSE, Chart)
                    # Split BSE price into price and percentage change
                    bse_price_parts = row_data[1].rsplit('-', 1)
                    bse_price = bse_price_parts[0]
                    bse_percentage_change = '-' + bse_price_parts[1] if len(bse_price_parts) > 1 else ''
                    # Split NSE price into price and percentage change
                    nse_price_parts = row_data[2].rsplit('-', 1)
                    nse_price = nse_price_parts[0]
                    nse_percentage_change = '-' + nse_price_parts[1] if len(nse_price_parts) > 1 else ''
                    # Update BSE and NSE columns with price and percentage change
                    if not bse_percentage_change:  # If BSE % change column is empty
                        # Extract last 4 characters from BSE price as percentage change
                        if bse_price[-4:][0]=='.':
                            bse_percentage_change = bse_price[-4:]
                            bse_price = bse_price[:-4]  # Remove last 4 characters from BSE price
                    row_data[1] = bse_price
                    row_data.insert(2, bse_percentage_change)
                    row_data[3] = nse_price
                    row_data.insert(4, nse_percentage_change)
                # Append row data to the list
                data.append(row_data)

            # Convert the data list to a pandas DataFrame
            df = pd.DataFrame(data, columns=['SCRIP', 'BSE PRICE(Rs)', 'BSE % CHANGE', 'NSE PRICE(Rs)', 'NSE % CHANGE', 'CHART', 'MORE INFO'])
            # Serialize the DataFrame to JSON format
            data_json = df.to_json(orient='records')

            # Extract <p> tags content
            p_tags = soup.find_all('p')
            p_contents = [p.get_text(strip=True) for p in p_tags]
            p_final = p_contents[2:6]
            print(p_final)

            # Send both the stock data and <p> tags content as JSON response
            return jsonify({'stock_data': data_json, 'p_contents': p_final})


            # Return the JSON data as a response
            # return jsonify(data_json)

    return jsonify({'error': 'Failed to fetch stock data'})


# @app.route('/fetch_financials', methods=['POST'])
# def fetch_financials():
#     competitors = request.form.getlist('competitors[]')

#     financial_details = []

#     for competitor in competitors:
#         try:
#             # Fetching financial data for the competitor
#             ticker_symbol = f"{competitor}.NS"  # Assuming competitors are Indian companies, adjust this accordingly
#             company = yf.Ticker(ticker_symbol)

#             # Fetch financial data
#             financials = company.financials
#             sales = financials.loc["Total Revenue"][-1]  # Last year's total revenue
#             profit = financials.loc["Net Income"][-1]  # Last year's net income
#             ebitda = financials.loc["EBITDA"][-1]  # Last year's EBITDA
#             gm = financials.loc["Gross Profit"][-1]  # Last year's gross profit

#             # Append financial details to the list
#             financial_details.append({'Competitor': competitor, 'Sales': sales, 'Profit': profit, 'EBITDA': ebitda, 'Gross Margin': gm})

#         except Exception as e:
#             print(f"Error fetching financial details for {competitor}: {str(e)}")

#     return jsonify(financial_details)
from datetime import datetime, timedelta

start_date = datetime.now() - timedelta(days=30)
# Global variable to store GNews object
gn_global = GNews(language='en', max_results=15, start_date=start_date)
gn_indian = GNews(language='en', max_results=15, country='IN',start_date=start_date)

# Function to fetch news based on selected query and GNews objectimport time  # Import the time module

def fetch_news(query, gn):
    try:
        # # Calculate the start date as today minus 30 days
        # start_date = datetime.now() - timedelta(days=30)
        
        # Fetch news articles
        time.sleep(1)  # Add a delay between requests for throttling
        articles = gn.get_news(query)
        
        if articles is not None and len(articles)>0:
            return articles
        else:
            print(f'No articles found for query: {query}')
            return []  # Return an empty list if no articles found

    except Exception as e:
        # Handle the exception gracefully
        print(f"An error occurred while fetching news: {e}")
        return []  # Return an empty list in case of error

import requests

@app.route('/fetch_project_news', methods=['POST'])
def fetch_project_news():
    
    # url = 'https://news.google.com'
    # response = requests.get(url, verify=False)  # Disable SSL verification
    # print(response.text)
    data = request.json  # Parse JSON data sent from frontend
    selected_news_types = data['news_types']  # Assuming the key is 'news_types'
    news_source = data.get('news_source', 'global')

    all_news = {}

    # Select GNews object based on news source
    gn = gn_global if news_source == 'global' else gn_indian
    print(news_source)

    # Define predefined categories
    predefined_categories = [
        'metal_commodities',
        'cement',
        'reinforcement steel',
        'crude oil',
        'diesel/petrol',
        'labor market',
        'currency exchange rates',
        'alternative energy sources',
        'construction machinery',
        'EV batteries',
        'logistics_supply_chain',
        'geopolitical_issues',
        'technology_news',
        'ESG in construction',
        'EHS news'
    ]

    for news_type in selected_news_types:
        if news_type == 'metal_commodities':
            articles = fetch_news('metal commodities news', gn)
            if articles:
                all_news['Metal Commodities'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
       
        elif news_type == 'cement':
            articles = fetch_news('cement news', gn)
            if articles:
                all_news['Cement'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]

        elif news_type == 'reinforcement steel':
            articles = fetch_news('reinforcement steel news in construction', gn)
            if articles:
                all_news['reinforcement steel'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]

        elif news_type == 'crude oil':
            articles = fetch_news('crude oil news', gn)
            if articles: 
                all_news['Crude Oil'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        elif news_type == 'diesel/petrol':
            articles = fetch_news('diesel petrol news', gn)
            if articles: 
                all_news['diesel/petrol'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        elif news_type == 'labor market':
            articles = fetch_news('labor market news', gn)
            if articles: 
                all_news['Labor Market'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        elif news_type == 'currency exchange rates':
            articles = fetch_news('currency exchange news', gn)
            if articles: 
                all_news['Currency Exchange Rates'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        elif news_type == 'alternative energy sources':
            articles = fetch_news('alternative energy sources news', gn)
            if articles: 
                all_news['Alternative Energy Sources'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        elif news_type == 'construction machinery':
            articles = fetch_news('construction machinery news', gn)
            if articles:
                all_news['Construction Machinery'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]

        elif news_type == 'EV batteries':
            articles = fetch_news('EV batteries news', gn)
            if articles: 
                all_news['EV batteries'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        

        elif news_type == 'logistics_supply_chain':
            articles = fetch_news('logistics & supply chain news', gn)
            if articles: 
                all_news['Logistics & Supply Chain'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        elif news_type == 'geopolitical_issues':
            articles = fetch_news('geopolitics issues news', gn)
            if articles:
                all_news['Geopolitical Issues'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        elif news_type == 'technology_news':
            articles = fetch_news('construction technology news', gn)
            if articles:
                all_news['Technology news'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        elif news_type == 'ESG in construction':
            articles = fetch_news('ESG in construction', gn)
            if articles:
                all_news['ESG in construction'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        elif news_type == 'EHS news':
            articles = fetch_news('EHS news', gn)
            if articles:
                all_news['EHS news'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
       # Check if there are any user-entered categories
    user_categories = [category for category in selected_news_types if category not in predefined_categories]
    print(user_categories[0])
    
    if user_categories[0]!='':
        for category in user_categories:
            articles = fetch_news(category, gn)
            all_news[category] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
            
    return jsonify(all_news) 
    # for news_type in selected_news_types:
    #     if news_type == 'metal_commodities':
    #         articles = fetch_news('metal commodities news')
    #         all_news['Metal Commodities'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
    #     elif news_type == 'cement':
    #         articles = fetch_news('cement news')
    #         all_news['Cement'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
    #     elif news_type == 'logistics_supply_chain':
    #         articles = fetch_news('logistics news')
    #         all_news['Logistics & Supply Chain'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
    #     elif news_type == 'geopolitical_issues':
    #         articles = fetch_news('geopolitical issues')
    #         all_news['Geopolitical Issues'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
    #     elif news_type == 'technology_news':
    #         articles = fetch_news('technology news')
    #         all_news['Technology news'] = [{'title': article['title'], 'Date': article['published date'], 'URL': article['url']} for article in articles]
        
        # Add more conditions for additional news types here

    # return jsonify(all_news)





@app.route('/fetch_articles', methods=['POST'])
def fetch_articles():

    
    sentence_similarity_threshold = 0.5
    word_similarity_threshold = 0.6
  
 
    start_date1= request.form['start_date']
    end_date1 = request.form['end_date']
    S_competitors = request.form.getlist('competitors[]')
    S_keywords = request.form.getlist('keywords[]')
    S_customers = request.form.getlist('customers[]')
    S_manufacturers = request.form.getlist('manufacturers[]')

    # start_date = datetime.datetime.strptime(start_date1, '%Y-%m-%d')
    # end_date = datetime.datetime.strptime(end_date1, '%Y-%m-%d')

    start_date = datetime.strptime(start_date1, '%Y-%m-%d')
    end_date = datetime.strptime(end_date1, '%Y-%m-%d')
   
    # parts = start_date1.split('-')
    # start_date = f'{parts[1]}/{parts[2]}/{parts[0]}'

    # parts = end_date1.split('-')
    # end_date = f'{parts[1]}/{parts[2]}/{parts[0]}'
       
    # Retrieve the manual competitors and keywords
    manual_competitor = request.form.get('manual_competitor[]')
    manual_keyword = request.form.get('manual_keyword[]')
    manual_customer = request.form.get('manual_customer[]')
    manual_manufacturer = request.form.get('manual_manufacturer[]')

    # Split the input string into a list
    competitors_list = manual_competitor.split(',')
    # Remove leading and trailing whitespaces from each item in the list
    competitors_list = [item.strip() for item in competitors_list]

    # Split the input string into a list
    keywords_list = manual_keyword.split(',')  
    # Remove leading and trailing whitespaces from each item in the list
    keywords_list = [item.strip() for item in keywords_list]
    # Add manual inputs to the respective lists if they are not empty

    # Split the input string into a list
    customers_list = manual_customer.split(',')  
    # Remove leading and trailing whitespaces from each item in the list
    customers_list = [item.strip() for item in customers_list]
    # Add manual inputs to the respective lists if they are not empty

    # Split the input string into a list
    manufacturers_list = manual_manufacturer.split(',')  
    # Remove leading and trailing whitespaces from each item in the list
    manufacturers_list = [item.strip() for item in manufacturers_list]
    # Add manual inputs to the respective lists if they are not empty

    if manual_competitor:
        for i in competitors_list:
            S_competitors.append(i)
    if manual_keyword:
        for i in keywords_list:
            S_keywords.append(i)
    if manual_customer:
        for i in customers_list:
            S_customers.append(i)
    if manual_manufacturer:
        for i in manufacturers_list:
            S_manufacturers.append(i) 

    S_competitors = list(set(S_competitors))
    S_customers = list(set(S_customers))
    S_manufacturers = list(set(S_manufacturers))

    print("Start Date:", start_date1)
    print("End Date:", end_date1)
    print("Competitors:", S_competitors)
    print("Customers:", S_customers)
    print("Manufacturers:", S_manufacturers)



    # gn = GoogleNews(lang='en',start=str(start_date),end=str(end_date))

    # Create an empty list to store article details
    article_details = []

    #adding all news titles in list
    titles = []

    # Create a set to track the processed URLs
    processed_urls = set()

   

    # if no keywords selected or entered in manual box, then prefixed keyword will run
    if S_keywords:
        S_keywords=list(set(S_keywords))
        print("Selected Keywords:", S_keywords)
    else:
        S_keywords=prefixed_keywords
        print("Prefixed Keywords:", S_keywords)



    # # Iterate over each competitor
    if S_competitors:
        for competitor in S_competitors: 
            # ticker_symbol = "ADANITRANS.NS"  # Replace with the company's ticker symbol
            # company = yf.Ticker(ticker_symbol)

            # financials = company.financials  # Fetch financial data
            # sales = financials.loc["Total Revenue"][-1]  # Last year's total revenue
            # profit = financials.loc["Net Income"][-1]  # Last year's net income

            # print("Last Year's Sales:", sales)
            # print("Last Year's Profit:", profit)
            print('1 Yaha pe hai')
            # gn = GoogleNews(lang='en',start=str(start_date),end=str(end_date))
            # gn.search(competitor.lower()) #will fetch top 10 news of that competitor

            # With this block of code using NewsAPI
            # newsapi_result = newsapi.get_everything(q=competitor.lower(), from_param=start_date1, to=end_date1, language='en', sort_by='publishedAt',page=2)
            try :

                gn = GNews(language='en',max_results=15,start_date=start_date,end_date=end_date)
                results = gn.get_news(competitor)

            except Exception as e:
                logging.error(f"Error checking similarity & processing article: {str(e)}")



            
            print('2 Yaha pe hai')
            # Add a delay after each API call
            time.sleep(delay_between_requests)


            # Fetch additional pages of search results
            # num_pages_to_fetch = 1  # Change this to the number of additional pages you want (e.g., 1 for a total of 2 pages)
            # for page in range(2, num_pages_to_fetch + 2):
            #     print('4 Ab yaha pe h')
            #     gn.get_page(page)
            #     # Add a delay after each additional API call
            #     time.sleep(delay_between_requests)
            
            count1=0
            for article in results:
                try:

                    print('length of articles fetched',len(results))
                    print('type of article',type(article))
                    title = article['title']
                    if title in titles:
                        print('Same title news found',title)
                        continue
                    else:
                        titles.append(title)
                    link = article['url']
                    date = article['published date']
                    try:
                        desc = gn.get_full_article(article['url'])
                        if desc is not None:
                            desc = desc.text
                        else:
                            desc = 'We were unable to scrape the description due to certain reasons'
                    except Exception as e:
                        print('Downloading article excp handled - ', e)
                        logging.error(f"Error downloading article: {str(e)}")
                    
                    publisher = article['publisher']
                    print('Publisher : ',publisher)
                    print('Content/Description :',desc)
                    print('Date : ',date)
                    keyword_mentioned = False
                    # Check if the article URL has already been processed
                    if link in processed_urls:
                        print('URL copy found----->')
                        continue  # Skip processing if the article has already been processed
                    else:
                        pass
                    count1+=1
                    print('1 loop ',count1,' news article---',competitor.upper(),'----->',title)
                    count=0
                    # Refine the keyword matching logic using regular expressions
                    for keyword in S_keywords:
                        
                        print('Keyword now is ---->',keyword)
                        
                        # lemmatized_keyword = lemmatizer.lemmatize(keyword.lower())  # Lemmatize the keyword

                        # # Lemmatize the title and description
                        # lemmatized_title = lemmatize_sentence(title.lower())
                        # lemmatized_desc = lemmatize_sentence(desc.lower())
                        # regex = r"\b" + re.escape(lemmatized_keyword) + r"\b"
                        # if re.search(regex, lemmatized_title, re.IGNORECASE) or re.search(regex, lemmatized_desc, re.IGNORECASE):
                        #     keyword_mentioned = True
                        #     print('1A loop---', competitor.upper(), '+', lemmatized_keyword.upper(), '----->', title)
                        #     count += 1
                        #     break
                        # else:
                        #     count += 1
                        #     print('Going to check', count, ' keyword in 1A')
                        
                    
                        if competitor == 'ReNew':
                            regex = r"\b" + re.escape(keyword) + r"\b"
                            if re.search(regex, title) or re.search(regex, desc):
                                keyword_mentioned = True
                                print('1A loop---',competitor.upper(),'+',keyword.upper(),'----->',title)
                                count+=1
                                break
                            else:
                                count+=1
                                print ('Going to check',count,' keyword in 1A')
                        else:
                            regex = r"\b" + re.escape(keyword) + r"\b"
                            if re.search(regex, title, re.IGNORECASE) or re.search(regex, desc, re.IGNORECASE):
                                keyword_mentioned = True
                                print('1A loop---',competitor.upper(),'+',keyword.upper(),'----->',title)
                                count+=1
                                break
                            else:
                                count+=1
                                print ('Going to check',count,' keyword in 1A')
                                
                except Exception as e:
                    logging.error(f"Error downloading article: {str(e)}")

                # Only consider articles that mention a keyword
                if keyword_mentioned:
                    try:
                        
                        # Alternative approach to Fetch the full content of the article mimicing a browser request
                        headers = {
                            'User-agent':
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582'
                        }
                        article = Article(link, headers=headers)
                        # article = Article(link)
                        article.download()

                        try :
                            article.parse()
                        except Exception as e:
                            pass

                        # Extract the article text
                        article_text=article.text

                        print('Article text is - ',article_text)
                        

                        # Check similarity with previously fetched articles
                        is_similar = False
                        try:
                            if len(article_details)>0:
                                for article in article_details:
                                
                                    if article['Competitor/Customer']==competitor:
                                        try:
                                        
                                            if (
                                                check_sentence_similarity(article['Description'],article_text, sentence_similarity_threshold) or
                                                check_word_similarity(article['Title'],title,  word_similarity_threshold)
                                            ):
                                                is_similar = True
                                                print('Similarity found, skipping it')                                     
                                                break
                                            else:
                                                pass
                                                
                                        except Exception as e:
                                            print('error is:',e)
                                            continue
                                            logging.error(f"Error checking similarity: {str(e)}")
                                    else:
                                        pass
                            else:
                                pass

                        except Exception as e:
                            logging.error(f"Error checking similarity & processing article: {str(e)}")
                            continue

                        if is_similar:
                            continue
                        else:
                            pass

                        
                        # Parse the text
                        try:
                            parser = PlaintextParser.from_string(desc, Tokenizer("english"))
                            print('Desc is','*'*15,desc)
                            print('parsing done in Competitor')
                            # # Summarize the text
                            summary1 = summarizer(parser.document, num_sentences)
                            summary = ' '.join([str(sentence) for sentence in summary1])
                            
                            # Summarize the text using the BERT-based summarizer
                            # summary = summarizer(article_text, ratio=0.2)  # Adjust the ratio as needed for the desired summary length
                            print('1A SUMMARY---->',summary)

                        except Exception as e:
                            logging.error(f"Error parsing article for summary: {str(e)}")
                            pass

                        
                        try:
                            # Extract keywords from the article content
                            rake = Rake()
                            rake.extract_keywords_from_text(desc)
                            keywords_extracted = rake.get_ranked_phrases()
                            print('Extracted keywords are ','*'*30,keywords_extracted)

                            if not keywords_extracted:
                                continue
                            else:
                                pass

                            if 'scrape' not in keywords_extracted and 'unable' not in keywords_extracted:
                                # # Generate word cloud from the extracted keywords
                                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords_extracted))

                                # # Generate the word cloud image
                                plt.figure(figsize=(8, 4))
                                plt.imshow(wordcloud, interpolation='bilinear')
                                plt.axis('off')

                                # # Convert the image to a base64 string
                                buffer = io.BytesIO()
                                plt.savefig(buffer, format='png')
                                buffer.seek(0)
                                image_base64 = base64.b64encode(buffer.read()).decode()
                                buffer.close()
                                # # Perform sentiment analysis on the article content
                                sia = SentimentIntensityAnalyzer()
                                sentiment_score = sia.polarity_scores(desc)['compound']

                                # Append article details to the list
                                article_details.append({'Title': title, 'URL': link, 'Date': date,'Sentiment_Score': sentiment_score,'Image_Base64': image_base64,
                                                    'Competitor/Customer': competitor,'Summary':summary,'Description':desc})
                            else:
                                
                                # Append article details to the list
                                article_details.append({'Title': title, 'URL': link, 'Date': date,
                                                    'Competitor/Customer': competitor,'Summary':summary,'Description':desc})

                            # # # Perform sentiment analysis on the article content
                            # sia = SentimentIntensityAnalyzer()
                            # sentiment_score = sia.polarity_scores(desc)['compound']

                            # # Append article details to the list
                            # article_details.append({'Title': title, 'URL': link, 'Date': date,'Sentiment_Score': sentiment_score,'Image_Base64': image_base64,
                            #                         'Competitor/Customer': competitor,'Summary':summary,'Description':desc})
                            # return 'appended'

                            # Add the processed URL to the set of processed URLs
                            processed_urls.add(link)
                            print("Links scraped uptill now",processed_urls)
                            # time.sleep(1)
                        
                        except Exception as e:
                            print('Error hai 1:',e)
                            logging.error(f"Error appending article: {str(e)}")
                            

                    except Exception as e:
                        print(f"Error processing article: {str(e)}")
                        logging.error(f"Error processing article: {str(e)}")
                        continue
              
                else:
                    continue
    else:
        pass


     # # Iterate over each competitor
    if S_customers:
        for customer in S_customers: 
            print('1 Yaha pe hai')
            # gn = GoogleNews(lang='en',start=str(start_date),end=str(end_date))
            # gn.search(competitor.lower()) #will fetch top 10 news of that competitor

            # With this block of code using NewsAPI
            # newsapi_result = newsapi.get_everything(q=competitor.lower(), from_param=start_date1, to=end_date1, language='en', sort_by='publishedAt',page=2)
            try :

                gn = GNews(language='en',max_results=15,start_date=start_date,end_date=end_date)
                results = gn.get_news(customer)

            except Exception as e:
                logging.error(f"Error checking similarity & processing article: {str(e)}")



            
            print('2 Yaha pe hai')
            # Add a delay after each API call
            time.sleep(delay_between_requests)


            # Fetch additional pages of search results
            # num_pages_to_fetch = 1  # Change this to the number of additional pages you want (e.g., 1 for a total of 2 pages)
            # for page in range(2, num_pages_to_fetch + 2):
            #     print('4 Ab yaha pe h')
            #     gn.get_page(page)
            #     # Add a delay after each additional API call
            #     time.sleep(delay_between_requests)
            
            count1=0
            for article in results:
                try:

                    print('length of articles fetched',len(results))
                    print('type of article',type(article))
                    title = article['title']
                    if title in titles:
                        print('Same title news found',title)
                        continue
                    else:
                        titles.append(title)
                    link = article['url']
                    date = article['published date']
                    try:
                        desc = gn.get_full_article(article['url'])
                        if desc is not None:
                            desc = desc.text
                        else:
                            desc = 'We were unable to scrape the description due to certain reasons'
                    except Exception as e:
                        print('Downloading article excp handled - ', e)
                        logging.error(f"Error downloading article: {str(e)}")
                    
                    publisher = article['publisher']
                    print('Publisher : ',publisher)
                    print('Content/Description :',desc)
                    print('Date : ',date)
                    keyword_mentioned = False
                    # Check if the article URL has already been processed
                    if link in processed_urls:
                        print('URL copy found----->')
                        continue  # Skip processing if the article has already been processed
                    else:
                        pass
                    count1+=1
                    print('1 loop ',count1,' news article---',customer.upper(),'----->',title)
                    count=0
                    # Refine the keyword matching logic using regular expressions
                    for keyword in S_keywords:
                        
                        print('Keyword now is ---->',keyword)
                        
                        # lemmatized_keyword = lemmatizer.lemmatize(keyword.lower())  # Lemmatize the keyword

                        # # Lemmatize the title and description
                        # lemmatized_title = lemmatize_sentence(title.lower())
                        # lemmatized_desc = lemmatize_sentence(desc.lower())
                        # regex = r"\b" + re.escape(lemmatized_keyword) + r"\b"
                        # if re.search(regex, lemmatized_title, re.IGNORECASE) or re.search(regex, lemmatized_desc, re.IGNORECASE):
                        #     keyword_mentioned = True
                        #     print('1A loop---', competitor.upper(), '+', lemmatized_keyword.upper(), '----->', title)
                        #     count += 1
                        #     break
                        # else:
                        #     count += 1
                        #     print('Going to check', count, ' keyword in 1A')
                        
                
                        regex = r"\b" + re.escape(keyword) + r"\b"
                        if re.search(regex, title, re.IGNORECASE) or re.search(regex, desc, re.IGNORECASE):
                            keyword_mentioned = True
                            print('1A loop---',customer.upper(),'+',keyword.upper(),'----->',title)
                            count+=1
                            break
                        else:
                            count+=1
                            print ('Going to check',count,' keyword in 1A')
                                
                except Exception as e:
                    logging.error(f"Error downloading article: {str(e)}")

                # Only consider articles that mention a keyword
                if keyword_mentioned:
                    try:
                        
                        # Alternative approach to Fetch the full content of the article mimicing a browser request
                        headers = {
                            'User-agent':
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582'
                        }
                        article = Article(link, headers=headers)
                        # article = Article(link)
                        article.download()

                        try :
                            article.parse()
                        except Exception as e:
                            pass

                        # Extract the article text
                        article_text=article.text

                        print('Article text is - ',article_text)
                        

                        # Check similarity with previously fetched articles
                        is_similar = False
                        try:
                            if len(article_details)>0:
                                for article in article_details:
                                
                                    if article['Competitor/Customer']==customer:
                                        try:
                                        
                                            if (
                                                check_sentence_similarity(article['Description'],article_text, sentence_similarity_threshold) or
                                                check_word_similarity(article['Title'],title,  word_similarity_threshold)
                                            ):
                                                is_similar = True
                                                print('Similarity found, skipping it')                                     
                                                break
                                            else:
                                                pass
                                                
                                        except Exception as e:
                                            print('error is:',e)
                                            continue
                                            logging.error(f"Error checking similarity: {str(e)}")

                                    else:
                                        pass
                            else:
                                pass

                        except Exception as e:
                            logging.error(f"Error checking similarity & processing article: {str(e)}")
                            continue

                        if is_similar:
                            continue
                        else:
                            pass

                        
                        # Parse the text
                        try:
                            parser = PlaintextParser.from_string(desc, Tokenizer("english"))
                            print('Desc is','*'*15,desc)
                            print('parsing done in Competitor')
                            # # Summarize the text
                            summary1 = summarizer(parser.document, num_sentences)
                            summary = ' '.join([str(sentence) for sentence in summary1])
                            
                            # Summarize the text using the BERT-based summarizer
                            # summary = summarizer(article_text, ratio=0.2)  # Adjust the ratio as needed for the desired summary length
                            print('1A SUMMARY---->',summary)

                        except Exception as e:
                            logging.error(f"Error parsing article for summary: {str(e)}")
                            pass

                        
                        try:
                            # Extract keywords from the article content
                            rake = Rake()
                            rake.extract_keywords_from_text(desc)
                            keywords_extracted = rake.get_ranked_phrases()
                            print('Extracted keywords are ','*'*30,keywords_extracted)

                            if not keywords_extracted:
                                continue
                            else:
                                pass

                            if 'scrape' not in keywords_extracted and 'unable' not in keywords_extracted:
                                # # Generate word cloud from the extracted keywords
                                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords_extracted))

                                # # Generate the word cloud image
                                plt.figure(figsize=(8, 4))
                                plt.imshow(wordcloud, interpolation='bilinear')
                                plt.axis('off')

                                # # Convert the image to a base64 string
                                buffer = io.BytesIO()
                                plt.savefig(buffer, format='png')
                                buffer.seek(0)
                                image_base64 = base64.b64encode(buffer.read()).decode()
                                buffer.close()
                                # # Perform sentiment analysis on the article content
                                sia = SentimentIntensityAnalyzer()
                                sentiment_score = sia.polarity_scores(desc)['compound']

                                # Append article details to the list
                                article_details.append({'Title': title, 'URL': link, 'Date': date,'Sentiment_Score': sentiment_score,'Image_Base64': image_base64,
                                                    'Competitor/Customer': customer,'Summary':summary,'Description':desc})
                            else:
                                
                                # Append article details to the list
                                article_details.append({'Title': title, 'URL': link, 'Date': date,
                                                    'Competitor/Customer': customer,'Summary':summary,'Description':desc})

                            # # # Perform sentiment analysis on the article content
                            # sia = SentimentIntensityAnalyzer()
                            # sentiment_score = sia.polarity_scores(desc)['compound']

                            # # Append article details to the list
                            # article_details.append({'Title': title, 'URL': link, 'Date': date,'Sentiment_Score': sentiment_score,'Image_Base64': image_base64,
                            #                         'Competitor/Customer': competitor,'Summary':summary,'Description':desc})
                            # return 'appended'

                            # Add the processed URL to the set of processed URLs
                            processed_urls.add(link)
                            print("Links scraped uptill now",processed_urls)
                            # time.sleep(1)
                        
                        except Exception as e:
                            print('Error hai 1:',e)
                            logging.error(f"Error appending article: {str(e)}")
                            

                    except Exception as e:
                        print(f"Error processing article: {str(e)}")
                        logging.error(f"Error processing article: {str(e)}")
                        continue
              
                else:
                    continue
    else:
        pass         

   
    # # Iterate over each competitor
    if S_manufacturers:
        for manufacturer in S_manufacturers: 
            print('now Yaha pe hai')
            # gn = GoogleNews(lang='en',start=str(start_date),end=str(end_date))
            # gn.search(competitor.lower()) #will fetch top 10 news of that competitor

            # With this block of code using NewsAPI
            # newsapi_result = newsapi.get_everything(q=competitor.lower(), from_param=start_date1, to=end_date1, language='en', sort_by='publishedAt',page=2)
            try :

                gn = GNews(language='en',max_results=15,start_date=start_date,end_date=end_date)
                results = gn.get_news(manufacturer)

            except Exception as e:
                logging.error(f"Error checking similarity & processing article: {str(e)}")



            
            print('2 Yaha pe hai')
            # Add a delay after each API call
            time.sleep(delay_between_requests)


            # Fetch additional pages of search results
            # num_pages_to_fetch = 1  # Change this to the number of additional pages you want (e.g., 1 for a total of 2 pages)
            # for page in range(2, num_pages_to_fetch + 2):
            #     print('4 Ab yaha pe h')
            #     gn.get_page(page)
            #     # Add a delay after each additional API call
            #     time.sleep(delay_between_requests)
            
            count1=0
            for article in results:
                try:

                    print('length of articles fetched',len(results))
                    print('type of article',type(article))
                    title = article['title']
                    if title in titles:
                        print('Same title news found',title)
                        continue
                    else:
                        titles.append(title)
                    link = article['url']
                    date = article['published date']
                    try:
                        desc = gn.get_full_article(article['url'])
                        if desc is not None:
                            desc = desc.text
                        else:
                            desc = 'We were unable to scrape the description due to SSL certification issues'
                    except Exception as e:
                        print('Downloading article excp handled - ', e)
                        logging.error(f"Error downloading article: {str(e)}")
                    
                    publisher = article['publisher']
                    print('Publisher : ',publisher)
                    print('Content/Description :',desc)
                    print('Date : ',date)
                    keyword_mentioned = False
                    # Check if the article URL has already been processed
                    if link in processed_urls:
                        print('URL copy found----->')
                        continue  # Skip processing if the article has already been processed
                    else:
                        pass
                    count1+=1
                    print('1 loop ',count1,' news article---',manufacturer.upper(),'----->',title)
                    count=0
                    # Refine the keyword matching logic using regular expressions
                    for keyword in S_keywords:
                        
                        print('Keyword now is ---->',keyword)
                        
                        # lemmatized_keyword = lemmatizer.lemmatize(keyword.lower())  # Lemmatize the keyword

                        # # Lemmatize the title and description
                        # lemmatized_title = lemmatize_sentence(title.lower())
                        # lemmatized_desc = lemmatize_sentence(desc.lower())
                        # regex = r"\b" + re.escape(lemmatized_keyword) + r"\b"
                        # if re.search(regex, lemmatized_title, re.IGNORECASE) or re.search(regex, lemmatized_desc, re.IGNORECASE):
                        #     keyword_mentioned = True
                        #     print('1A loop---', competitor.upper(), '+', lemmatized_keyword.upper(), '----->', title)
                        #     count += 1
                        #     break
                        # else:
                        #     count += 1
                        #     print('Going to check', count, ' keyword in 1A')
                        
                    
                    
                        regex = r"\b" + re.escape(keyword) + r"\b"
                        if re.search(regex, title, re.IGNORECASE) or re.search(regex, desc, re.IGNORECASE):
                            keyword_mentioned = True
                            print('1A loop---',manufacturer.upper(),'+',keyword.upper(),'----->',title)
                            count+=1
                            break
                        else:
                            count+=1
                            print ('Going to check',count,' keyword in 1A')
                                
                except Exception as e:
                    logging.error(f"Error downloading article: {str(e)}")

                # Only consider articles that mention a keyword
                if keyword_mentioned:
                    try:
                        
                        # Alternative approach to Fetch the full content of the article mimicing a browser request
                        headers = {
                            'User-agent':
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582'
                        }
                        article = Article(link, headers=headers)
                        # article = Article(link)
                        article.download()

                        try :
                            article.parse()
                        except Exception as e:
                            pass

                        # Extract the article text
                        article_text=article.text

                        print('Article text is - ',article_text)
                        

                        # Check similarity with previously fetched articles
                        is_similar = False
                        try:
                            if len(article_details)>0:
                                for article in article_details:
                                
                                    if article['Competitor/Customer']==manufacturer:
                                        try:
                                        
                                            if (
                                                check_sentence_similarity(article['Description'],article_text, sentence_similarity_threshold) or
                                                check_word_similarity(article['Title'],title,  word_similarity_threshold)
                                            ):
                                                is_similar = True
                                                print('Similarity found, skipping it')                                     
                                                break
                                            else:
                                                pass
                                                
                                        except Exception as e:
                                            print('error is:',e)
                                            continue
                                            logging.error(f"Error checking similarity: {str(e)}")
                                    else:
                                        pass
                            else:
                                pass

                        except Exception as e:
                            logging.error(f"Error checking similarity & processing article: {str(e)}")
                            continue

                        if is_similar:
                            continue
                        else:
                            pass

                        
                        # Parse the text
                        # Parse the text
                        try:
                            parser = PlaintextParser.from_string(desc, Tokenizer("english"))
                            print('Desc is','*'*15,desc)
                            print('parsing done in Competitor')
                            # # Summarize the text
                            summary1 = summarizer(parser.document, num_sentences)
                            summary = ' '.join([str(sentence) for sentence in summary1])
                            
                            # Summarize the text using the BERT-based summarizer
                            # summary = summarizer(article_text, ratio=0.2)  # Adjust the ratio as needed for the desired summary length
                            print('1A SUMMARY---->',summary)

                        except Exception as e:
                            logging.error(f"Error parsing article for summary: {str(e)}")
                            pass

                        
                        try:
                            # Extract keywords from the article content
                            rake = Rake()
                            rake.extract_keywords_from_text(desc)
                            keywords_extracted = rake.get_ranked_phrases()
                            print('Extracted keywords are ','*'*30,keywords_extracted)

                            if not keywords_extracted:
                                continue
                            else:
                                pass

                            if 'scrape' not in keywords_extracted and 'unable' not in keywords_extracted:
                                # # Generate word cloud from the extracted keywords
                                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords_extracted))

                                # # Generate the word cloud image
                                plt.figure(figsize=(8, 4))
                                plt.imshow(wordcloud, interpolation='bilinear')
                                plt.axis('off')

                                # # Convert the image to a base64 string
                                buffer = io.BytesIO()
                                plt.savefig(buffer, format='png')
                                buffer.seek(0)
                                image_base64 = base64.b64encode(buffer.read()).decode()
                                buffer.close()
                                # # Perform sentiment analysis on the article content
                                sia = SentimentIntensityAnalyzer()
                                sentiment_score = sia.polarity_scores(desc)['compound']

                                # Append article details to the list
                                article_details.append({'Title': title, 'URL': link, 'Date': date,'Sentiment_Score': sentiment_score,'Image_Base64': image_base64,
                                                    'Competitor/Customer': manufacturer,'Summary':summary,'Description':desc})
                            else:
                                
                                # Append article details to the list
                                article_details.append({'Title': title, 'URL': link, 'Date': date,
                                                    'Competitor/Customer': manufacturer,'Summary':summary,'Description':desc})

                            # # # Perform sentiment analysis on the article content
                            # sia = SentimentIntensityAnalyzer()
                            # sentiment_score = sia.polarity_scores(desc)['compound']

                            # # Append article details to the list
                            # article_details.append({'Title': title, 'URL': link, 'Date': date,'Sentiment_Score': sentiment_score,'Image_Base64': image_base64,
                            #                         'Competitor/Customer': competitor,'Summary':summary,'Description':desc})
                            # return 'appended'

                            # Add the processed URL to the set of processed URLs
                            processed_urls.add(link)
                            print("Links scraped uptill now",processed_urls)
                            # time.sleep(1)
                        
                        except Exception as e:
                            print('Error hai 1:',e)
                            logging.error(f"Error appending article: {str(e)}")
                            

                    except Exception as e:
                        print(f"Error processing article: {str(e)}")
                        logging.error(f"Error processing article: {str(e)}")
                        continue
              
                else:
                    continue
    else:
        pass
    
    # Create a dataframe from the collected article details
    df = pd.DataFrame(article_details, columns=['Title', 'URL', 'Date', 'Sentiment_Score','Image_Base64','Competitor/Customer','Summary','Description'])
    desired_columns = ['Title', 'URL', 'Date', 'Competitor/Customer','Summary','Description']
    df[desired_columns].to_csv('PTDnews1.csv',index=False)

    # Create a list of articles
    global articles
    articles = df.to_dict(orient='records')
    
    
   
    return render_template('articles.html', articles=articles)




@app.route('/summarized.html')
def summarized():
    # Perform backend operations and retrieve necessary details
   
    articles_by_competitor = {}
    titles_by_competitor = {}
    # return 'chlra h'
    for article in articles:
        
        # print('article in articles is :',article)
        competitor = article['Competitor/Customer']
        if competitor not in articles_by_competitor:
            articles_by_competitor[competitor] = []
        articles_by_competitor[competitor].append(article['Summary'])
        print('articles by comp',articles_by_competitor)

    for article in articles:
        # print('article in articles is :',article)
        competitor = article['Competitor/Customer']
        if competitor not in titles_by_competitor:
            titles_by_competitor[competitor] = []
        titles_by_competitor[competitor].append(article['Title'])
        print('titles by comp',titles_by_competitor)
    
    # Generate summarized news for each competitor
    competitor_summaries = {}
    titles_summaries = {}
    for competitor, competitor_articles in articles_by_competitor.items():
        # print('competitor is : ',competitor)
        # print('article is : ',competitor_articles)
        article_corpus = ''
        article_corpus += ''.join([str(i) for i in competitor_articles])
        print('article corpus is :',article_corpus)
        parser = PlaintextParser.from_string(article_corpus, Tokenizer("english"))
        # Summarize the text
        summary1 = summarizer(parser.document, num_sentences1)
        print('summary1: ',summary1)
        summarized_news = ''.join([str(sentence) for sentence in summary1])
        # summarized_news = generate_summary(article_corpus)  # Implement your summarization logic here
        
        competitor_summaries[competitor] = summarized_news


    for competitor, competitor_titles in titles_by_competitor.items():
        title_corpus = ''
        title_corpus += '...'.join([str(i) for i in competitor_titles])
        print('title corpus is :',title_corpus)
        summarized_news = title_corpus

        titles_summaries[competitor] = summarized_news


    # Render the 'page2.html' template and pass the data to it
    return render_template('summarized.html', data=competitor_summaries,data1=titles_summaries)



from flask import Flask, render_template, request, jsonify
import feedparser



@app.route('/fetch_global_news', methods=['GET'])
def fetch_global_news():

    websites = [
        # 'www.energynow.com',
        'www.solarpowerworldonline.com',
        'www.africa-energy.com',
        'www.esi-africa.com',
        'www.powerengineeringint.com',
        # 'www.engineeringnews.co.za',
        # 'www.miningweekly.com',
        'www.power-eng.com',
        # 'www.utilitydive.com',
        # 'www.greentechmedia.com',
        # 'www.eenews.net',
        'www.renewableenergyworld.com',
        # 'www.power-technology.com',
        # 'www.energetica-india.net',
        # 'www.euroelectric.org',
        'www.windpowermonthly.com',
        # 'www.energyworldmag.com',
        # 'www.powerrussia.ru',
        # 'www.electricityjournal.com',
        # 'www.neftegaz.ru',
        # 'www.energyland.info',
        # 'www.asiapowerweek.com',
        # 'www.appeec.org',
        # 'www.energycentral.com',
        # 'www.eai.in',
        # 'www.theenergycollective.com',
        'www.powermag.com',
        'www.power-grid.com',
        'www.world-nuclear-news.org',
        # 'www.energy.gov',
        # 'www.nrel.gov',
        # 'www.iea.org',
        # 'www.iaea.org',
        # 'www.elp.com',
        # 'www.energyinfrapost.com',
        # 'www.smartgridnews.com',
        # 'www.utilityproducts.com',
        # 'www.ogj.com',
        # 'www.energy-storage.news',
        'www.pv-magazine.com'
    ]

    websites = list(set(websites))

    keywords = ['renewable','offshore wind','power sector','energy','power transmission',' electricity transmission', 'transmission & distribution', 'solar energy', 'solar plant', 'renewable energy', 'green energy', 'carbon footprint']

    all_relevant_articles = []

    for website in websites:
        print(f"Fetching RSS feeds from {website}")
        feed_url = f"https://{website}/rss"
        feed = feedparser.parse(feed_url)

        relevant_articles = []
        for entry in feed.entries:
            entry_title_lower = entry.title.lower()
            entry_summary_lower = getattr(entry, 'summary', '').lower()  # Check if 'summary' exists in entry

            for keyword in keywords:
                if keyword in entry_title_lower or keyword in entry_summary_lower:
                    relevant_articles.append({
                        'title': entry.title,
                        'published_date': entry.published,
                        'summary': entry_summary_lower  # Store the lowercased summary
                    })
                    break

        print(f"Found {len(relevant_articles)} relevant articles from {website}")
        all_relevant_articles.extend(relevant_articles)

    return jsonify(all_relevant_articles)






from gnews import GNews
# from googletrans import Translatorpio
from translate import Translator
import concurrent.futures
from datetime import datetime, timedelta
import time
import logging
from re import search

@app.route('/fetch_LT_news', methods=['GET'])
def fetch_LT_news():
    # Define translations
    # translations = {
    #     # 'en': ['L&T', 'Lnt', 'LandT','Larsen & Toubro','L and T'],
    #     'mr': ['एल आणि टी', 'एलएनटी', 'लँडटी', 'लर्सन आणि टूब्रो', 'एल अँड टी'],
    #     'hi': ['एल एंड टी', 'एलएनटी', 'लैंडटी', 'लर्सन एंड टूब्रो', 'एल एंड टी'],
    #     'bn': ['এল এবং টি', 'এলএনটি', 'ল্যান্ডটি', 'লারসেন এবং টুব্রো', 'এল এবং টি'],
    #     'ta': ['எல் ஆன்ட் டி', 'எல் என் டி', 'லேண்டி', 'லார்சன் ஆன்ட் டூப்ரோ','லார்சன் மற்றும் டூப்ரோ' ,'எல் ஆன்ட் டி'],
    #     'te': ['ఎల్ అండ్ టి', 'ఎల్ఎన్టి', 'ల్యాండ్టి', 'లార్సెన్ అండ్ టూబ్రో', 'ఎల్ అండ్ టి'],
    #     'ml': ['എല്‍ ആന്റ് ടി', 'എല്‍എന്റി', 'ലാന്ഡ്റ്റി', 'ലാര്‍സണ്‍ ആന്റ് ടൂബ്രോ', 'എല്‍ ആന്റ് ടി']
    # }

    translations = {
        'en': ['Larsen and Toubro'],
        # 'mr': ['एल अँड टी','एलएनटी'],
        # 'hi': ['एल एंड टी','एलएनटी','लर्सन एंड टूब्रो','एल अँड ट'],
        # 'bn': ['লারসেন এবং টুব্রো','লারসেন অ্যান্ড টুব্রো'],
        # 'ta': ['எல் மற்றும் டி', 'லார்சன் மற்றும் டூப்ரோ'],
        # 'te': ['ఎల్ మరియు టి', 'లార్సెన్ మరియు టూబ్రో'],
        # 'ml': ['എൽ, ടി', 'ലാർസെൻ ആൻഡ് ടൂബ്രോ'],
        # 'or': ['ଲାରସେନ ଏଣ୍ଡ ଟାଉବ୍ରୋ', 'ଏଲ ଏଣ୍ଡ ଟି'],
        # 'gu': ['લાર્સન અને ટુબ્રો', 'એલ અને ટી']
    }

    # for keys,values in translations.items():
    #     for value in values:
    #         print(value)

    # Create an empty list to store article details
    article_details = []

    #adding all news titles in list
    # titles = []

    # Create a set to track the processed URLs
    processed_urls = set()

    # Introduce a delay of 2 seconds between each API call
    delay_between_requests = 1

    # translator = Translator(to_lang='en')
    # Calculate the start date as 1 week before today
    start_date = datetime.now() - timedelta(days=30)

    # End date remains as today
    end_date = datetime.now()

    # Formatting the dates to match the given format
    start_date_formatted = start_date.strftime('%Y-%m-%d')
    end_date_formatted = end_date.strftime('%Y-%m-%d')

    print("Start Date:", start_date_formatted)
    print("End Date:", end_date_formatted)

    start_date = datetime.strptime(start_date_formatted, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_formatted, '%Y-%m-%d')

    count=0
    for lang, terms in translations.items():
        for term in terms: 

            print(f'This term is {term}, language is {lang}')
            gn = GNews(language=lang,max_results=30,start_date=start_date)
            results = gn.get_news(term)
            time.sleep(delay_between_requests)
            print('length of articles fetched',len(results))

            
            for article in results:
                        
                    try:
                        
                        # print('length of articles fetched',len(results))
                        # print('type of article',type(article))
                        title = article['title']
                        # # Perform sentiment analysis on the article content
                        sia = SentimentIntensityAnalyzer()
                        sentiment_score = sia.polarity_scores(title)['compound']
                        link = article['url']
                        # Check if the article URL has already been processed
                        if link in processed_urls:
                            print('URL copy found----->')
                            continue  # Skip processing if the article has already been processed
                        else:
                            pass
                        processed_urls.add(link)

                        # translated_title = translator.translate(title)
        
                        # # Print the translated title
                        # print(f"Translated Title ({lang} to English): {translated_title}")
                        
                        # if translated_title in titles:
                        #     print('Same title news found',title)
                        #     continue
                        # else:
                        #     titles.append(translated_title)
                        
                        date = article['published date']
                        # try:
                        #     desc = gn.get_full_article(article['url'])
                        #     if desc is not None:
                        #         desc = desc.text
                        #     else:
                        #         desc = 'We were unable to scrape the description due to certain reasons'
                        # except Exception as e:
                        #     print('Downloading article excp handled - ', e)
                        #     logging.error(f"Error processing article: {str(e)}")
                        #     pass
                        
                        publisher = article['publisher']
                        print('Publisher : ',publisher)
                        # print('Content/Description :',desc)
                        print('Date : ',date)
                        # is_relevant = False
                        print('Title',title)
                        

                        # # Check for keyword matches in title or description
                        # if any(re.search(r'\b' + re.escape(keyword) + r'\b', title) for keyword in terms) or any(re.search(r'\b' + re.escape(keyword) + r'\b', desc) for keyword in terms):
                        #     is_relevant = True

                        # if not is_relevant:
                        #     continue  # Skip irrelevant articles

                        # Add the processed URL to the set of processed URLs
                        
                        # Append article details to the list
                        article_details.append({'Title': title, 'URL': link, 'Date': date,
                                                'Publisher': publisher,'Language':lang,'sentiment_score':sentiment_score})
                        count+=1
                        print(f'{count} loop running, keyword is {term}, & language is {lang}')

                    except Exception as e:
                        logging.error(f"Error processing article: {str(e)}")
                        print(f"Error processing article: {str(e)}")
                        pass

    return jsonify(article_details)



if __name__ == '__main__':
    logging.basicConfig(filename='error1.log', level=logging.ERROR)
    app.run(debug=True)



