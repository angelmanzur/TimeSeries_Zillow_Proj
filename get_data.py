import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def get_the_data():

	df = pd.read_csv('zillow_data.csv')

	# get the DC data only
	dc_df = df[df['State']=='DC']

	# drop unused columns
	dc_df.drop(['RegionID','City','Metro','CountyName','SizeRank','State'],
           axis=1, inplace=True)

	# get the list of zipcodes 
	zipcodes = list(dc_df['RegionName'])
	dc_df.drop(['RegionName'],axis=1,inplace=True)
	dc_df.head()

	dc_df_T = dc_df.transpose(copy=True)
	dc_df_T.reset_index()
	dc_df_T['date'] = pd.to_datetime(dc_df_T.index)
	dc_df_T.set_index('date', inplace=True)
	dc_df_T.columns = zipcodes

	return dc_df_T

