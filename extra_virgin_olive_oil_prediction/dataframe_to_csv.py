import requests
import json
import pandas as pd
import matplotlib.pyplot as plt


# take json data from API
response = requests.get("http://148.251.22.254:8080/price-api-1.0/price/findAll")
data = response.text
parsed = json.loads(data)
# print(json.dumps(parsed, indent=4))

# dataframe
df = pd.DataFrame(parsed)
prouped_ordered = df.groupby(['product']).size().reset_index(name='counts').sort_values('counts')
print(prouped_ordered)

# print(dfevoo)
# print(prouped_ordered.head(250))

dfevoo = df[df['product'] == 'μπανάνες']
dfevoo = df[df['product'].str.contains("olive")]
dfevoo = dfevoo[dfevoo['country'] == 'greece']

dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    by='priceStringDate')

dfevoo = pd.DataFrame(dfevoo)
# dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
print(dfevoo)
data=dfevoo['price']
#Data Visualization
plt.figure(figsize = (18,9))
plt.plot(dfevoo['price'])
plt.xlabel('priceStringDate',fontsize=18)
plt.ylabel('price',fontsize=18)
plt.show()
# quit(0)
quit(0)

#df.to_csv('food_dataset.csv', header=False, index=False)
#df.to_csv('food_dataset.csv', sep='\t', encoding='utf-8')
df.to_csv('food_dataset.csv', index=False)