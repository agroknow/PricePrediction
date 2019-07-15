import sqlite3
conn = sqlite3.connect('Train_results.db')

c = conn.cursor()

dropTableStatement = "DROP TABLE TrainingResults"

c.execute(dropTableStatement)

c.execute(('''CREATE TABLE TrainingResults
         (ID INT AUTOINCREMENT PRIMARY KEY      NOT NULL,
         Product_name           TEXT    NOT NULL,
         Country_name           TEXT     NOT NULL,
         Data_Source        TEXT NOT NULL,
         Model_fit         BLOB);'''))

# c.execute('''
#    INSERT INTO TrainingResults(ID,Product_Name,Country_Name,Data_Source,Model_fit)
#    VALUES (2,'product','country','data_Source','fitted_model')
#
#              ''')
#
# c.execute('''
# SELECT *
# FROM TrainingResults
#           ''')
# print(c.fetchall())
#
# ID=3
product="maria"
country='maria'
data_Source= 'maria'
fitted_model=44
# c.execute('''
#    INSERT INTO TrainingResults(ID,Product_Name,Country_Name,Data_Source,Model_fit)
#     VALUES (?,?,?,?,?) ''',(ID,product,country,data_Source,fitted_model))
#
# c.execute('''
# SELECT *
# FROM TrainingResults
#           ''')
# print(c.fetchall())

# c.execute('''
#       INSERT INTO TrainingResults(ID,Product_Name,Country_Name,Data_Source,Model_fit)
#       VALUES (?,?,?,?,?) ''', (1, product, country, data_Source, open('model_fit.pkl', 'rb').read()))
# c.execute('''
#       INSERT INTO TrainingResults(ID,Product_Name,Country_Name,Data_Source,Model_fit)
#       VALUES (?,?,?,?,?) ''', (2, product, country, data_Source, fitted_model))
#
# c.execute('''
#    SELECT *
#    FROM TrainingResults
#            ''')
# conn.commit()
# print(c.fetchall())
# c.close()
# quit(0)