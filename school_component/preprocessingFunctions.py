import pandas
import difflib

def excludeNegatives(df):
	"""this function excludes schools for which desired data is not available.  in the datasets, 
	this is indicated with a value of -1"""

	for col in df.columns:
		if col != 'classifier':
			df = df[ df[col] >= 0 ]
			df = df[ df[col].notnull() ]
	return df

def standardize(df):
	"""support vector machines require standardized input or standardized kernels.  this
	function standardizes the appropriate columns of input data"""

	st = pandas.DataFrame()
	for col in df.columns:
		if df[col].dtype.kind == 'f':
			st[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
		else:
			st[col] = df[col]
	return st

def addFinancialData(dat):
	"""funding data is available only by district.  this function matches the district data to 
	the appropriate row of campus data using a 'close match' string function, and returns the 
	merged dataframe"""

	#read in datasets
	fin = pandas.read_csv('ELSI.csv')
	dat['match'] = ''

	#find closest matching district name (repeated three times for accurate matching)
	no_matches = []
	multiple_matches = []
	for i, row in dat.iterrows():
		if row.match == '':
			new_name = difflib.get_close_matches(row.district_name, fin.Name.values, cutoff=.9)
			if len(new_name) == 0:
				no_matches.append(row.district_name)
			elif len(new_name) > 1:
				multiple_matches.append(row.district_name)
				dat.loc[ dat.district_name == row.district_name, 'match'] = new_name[0]
				fin = fin[ fin.Name != new_name[0] ]
			else:
				dat.loc[ dat.district_name == row.district_name, 'match'] = new_name[0]
				fin = fin[ fin.Name != new_name[0] ]

	no_matches = []
	multiple_matches = []
	for i, row in dat.iterrows():
		if row.match == '':
			new_name = difflib.get_close_matches(row.district_name, fin.Name.values, cutoff=.8)
			if len(new_name) == 0:
				no_matches.append(row.district_name)
			elif len(new_name) > 1:
				multiple_matches.append(row.district_name)
				dat.loc[ dat.district_name == row.district_name, 'match'] = new_name[0]
				fin = fin[ fin.Name != new_name[0] ]
			else:
				dat.loc[ dat.district_name == row.district_name, 'match'] = new_name[0]
				fin = fin[ fin.Name != new_name[0] ]

	no_matches = []
	multiple_matches = []
	for i, row in dat.iterrows():
		if row.match == '':
			new_name = difflib.get_close_matches(row.district_name, fin.Name.values, cutoff=.7)
			if len(new_name) == 0:
				no_matches.append(row.district_name)
			elif len(new_name) > 1:
				multiple_matches.append(row.district_name)
				dat.loc[ dat.district_name == row.district_name, 'match'] = new_name[0]
				fin = fin[ fin.Name != new_name[0] ]
			else:
				dat.loc[ dat.district_name == row.district_name, 'match'] = new_name[0]
				fin = fin[ fin.Name != new_name[0] ]

	#match the one campus that didn't find a match
	dat.loc[ dat.district_name == 'ROSCOE COLLEGIATE ISD', 'match'] = 'ROSCOE ISD, TX'

	#merge datasets based on match
	fin2 = pandas.read_csv('ELSI.csv')
	combined = pandas.merge(dat, fin2, left_on='match', right_on='Name', how='left')

	#do some final cleaning of the data
	combined = combined.drop('match', 1)
	combined = combined.drop('Name', 1)
	combined[['total_rev', 't_fed_rev', 't_st_rev', 't_loc_rev']] = combined[['total_rev', 't_fed_rev', 't_st_rev', 't_loc_rev']].astype(float)
	combined.total_rev = combined.total_rev*combined.studentCount
	combined.t_fed_rev = combined.t_fed_rev*combined.studentCount
	combined.t_st_rev = combined.t_st_rev*combined.studentCount
	combined.t_loc_rev = combined.t_loc_rev*combined.studentCount

	return combined


