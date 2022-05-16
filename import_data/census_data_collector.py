import requests
import pandas as pd

# Set census state code to Illinois
STATE = 17

class CensusQuery():

    api_key = "83fdc3ae9f205e7d930e9b92321c489fbcc4707e"

    def __init__(self):
        '''
        Construct a Census API query class. 
        '''
        self.dataset_topic = None
        self.retrieval_error = False
        self.query = None


    def retrieve_data(self, year=None, dataset=None, var_lst=None, unit=None,
                    dataset_topic=None):
        '''
        Method to retrieve data for Illinois from the Census Bureau's API.

        Input:
            year : (int) year of data
            dataset : (str) dataset name
            var_lst : (list) list of variables
            unit : (str) geographical unit the data has been taken from
        
        Output:
            pandas dataframe (returns None if request is bad)
        '''
        
        # Build query
        if not self.query:
            self.dataset_topic = dataset_topic
            query_base = f"https://api.census.gov/data/{year}/{dataset}"
            query_vars = f"?get={','.join(var_lst)}"
            query_geo = f"&for={unit}:*&in=state:{STATE}"
            key = f"&key={CensusQuery.api_key}"
            self.query = query_base + query_vars + query_geo + key

        # Query API
        response = requests.get(self.query)
        if not response.status_code == requests.codes.ok:
                self.retrieval_error = True
                return None
    
        print(f'Retrieved {dataset_topic} data from Census Bureau API.')

        # Build dataframe
        data_json = response.json()
        census_df = pd.DataFrame(data_json[1:], columns=data_json[0])
        census_df = census_df.drop('state', axis=1)
        
        return census_df


    def retry_retrieval(self):
        '''
        Method to reattmept data query.

        Input:
            None
        
        Output
            pandas dataframe (returns None if request is bad)
        '''
        if not self.query:
            print('No query has yet been made to re-attempt.')
            return None
        census_df = self.retrieve_data()

        while not census_df:
            another_retry = input(f'The {self.dataset_topic} dataset query has'
            'returned an error code, attempt again? (y or n):')
            if another_retry == 'y' or another_retry == 'Y':
                census_df = self.retrieve_data()
            else:
                print(f'Continuing without {self.dataset_topic} dataset.')
                return None
        
        self.retrieval_error = False
        return census_df


    def __repr__(self):
        '''
        Repr method for CensusQuery class.
        '''
        if not self.dataset_topic:
            print('No query has yet been made with this CensusQuery instance.')
        else:
            print(f'Query topic: Census {self.dataset_topic} dataset\n'
                f'Query: {self.query}\n'
                f'Retrieval error: {self.retrieval_error}')