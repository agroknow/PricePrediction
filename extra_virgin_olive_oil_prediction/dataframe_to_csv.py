import requests
import json
import pandas as pd


# take json data from API
response = requests.get("http://148.251.22.254:8080/price-api-1.0/price/findAll")
data = response.text
parsed = json.loads(data)
# print(json.dumps(parsed, indent=4))

# dataframe
df = pd.DataFrame(parsed)

#df.to_csv('food_dataset.csv', header=False, index=False)
#df.to_csv('food_dataset.csv', sep='\t', encoding='utf-8')
df.to_csv('food_dataset.csv', index=False)