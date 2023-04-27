import requests

URL = "https://ca.finance.yahoo.com/"
page = requests.get(URL)

print(page.text)