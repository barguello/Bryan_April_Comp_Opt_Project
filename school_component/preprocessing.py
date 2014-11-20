import pandas
import preprocessingFunctions

#read in key file
reference = pandas.read_csv('data_description.csv')
df = pandas.DataFrame()

#read in interesting features
for ds in reference.dataset.unique():

	data = pandas.read_csv(str(ds)+'.csv')
	temp = pandas.DataFrame()

	#for each feature from this dataset:	
	for i, row in reference[ reference.dataset == ds ].iterrows():
		temp[row.feature] = data[row.code]
		temp[row.feature] = temp[row.feature].convert_objects(convert_numeric = True)

	#merge into master df
	if df.empty:
		df = temp
	else:
		df = pandas.merge(df, temp, on='campus', how='left')


#sort out schools not rated 'M' or 'I', charter schools, and AEC schools
df = df[ (df.rating == 'M') | (df.rating == 'I')]
df = df[ df.charter == 'N' ]
df = df[ df.aec == 'N' ]

#add classifier column
df['classifier'] = -1
df.loc[ df.rating == 'M', 'classifier'] = 1

#create high school dataframe
hs = pandas.DataFrame()
hs = df[ df.type == 'S' ]
hs = preprocessingFunctions.excludeNegatives(hs)

#add financial data
hs = preprocessingFunctions.addFinancialData(hs)

#create standardized high school dataframe
hs = preprocessingFunctions.standardize(hs)
hs.to_csv('hs.csv', index=False)

#create feature list
feature_list = list(hs.columns[12:])
feature_list.remove('classifier')
feature_list.remove('studentCount')
print feature_list







