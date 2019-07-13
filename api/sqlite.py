import sqlite3
conn = sqlite3.connect('Train_results.db')

c = conn.cursor()
c.execute('''
   INSERT INTO TrainingResults(Product_Name,Country_Name,DataSource,Model_fit)
   VALUES ('product','country','data_Source','fitted_model')

             ''')

c.execute('''
SELECT *
FROM TrainingResults 
          ''')



product="maria"
country='maria'
data_Source= 'maria'
fitted_model=44
c.execute('''
   INSERT INTO TrainingResults(Product_Name,Country_Name,DataSource,Model_fit)
   VALUES (?,?,?,?) ''',(product,country,data_Source,fitted_model))

c.execute('''
SELECT *
FROM TrainingResults 
          ''')
print(c.fetchall())
quit(0)