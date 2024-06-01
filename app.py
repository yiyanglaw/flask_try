from flask import Flask, request
import pandas as pd

app = Flask(__name__)

# Load the data
data = pd.read_csv('p2.csv')
phishing_urls = data['URL'].tolist()

@app.route('/check_url', methods=['POST'])
def check_url():
    # Get the user input URL from the request
    url = request.form['url']

    # Check if the URL is similar to any of the phishing URLs
    for phishing_url in phishing_urls:
        if phishing_url in url:
            return 'True'

    # If no match is found, return False
    return 'False'

if __name__ == '__main__':
    app.run(host='0.0.0.0')