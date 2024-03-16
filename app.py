from flask import Flask, render_template
from GoogleNews import GoogleNews
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    # Create a GoogleNews instance
    googlenews = GoogleNews()

    # Set search query for L&T
    googlenews.search('L&T')

    # Set the time period
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 2, 1)

    # Format the time period for the search query
    time_period = f'{start_date.strftime("%m/%d/%Y")} - {end_date.strftime("%m/%d/%Y")}'

    # Set the time period in the search query
    googlenews.time_range(time_period)

    # Get news articles
    articles = googlenews.get_texts()

    # Pass the articles to the template
    return render_template('index.html', articles=articles)

if __name__ == '__main__':
    app.run(debug=True)
