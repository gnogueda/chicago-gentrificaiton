import requests
import pandas as pd

# Set census state code to Illinois
STATE = 17

# Set zip codes to those within Cook county
ZIP_CODES = ['60104', '60126', '60130', '60131', '60141', '60153', '60154', '60155', '60160', '60161', '60162', '60163', '60164', '60165', '60171', '60176', '60301', '60302', '60303', '60304', '60305', '60402', '60455', '60457', '60458', '60459', '60480', '60499', '60501', '60513', '60521', '60523', '60525', '60526', '60527', '60534', '60546', '60558', '60601', '60602', '60603', '60604', '60605', '60606', '60607', '60608', '60610', '60611', '60612', '60613', '60614', '60615', '60616', '60618', '60622', '60623', '60624', '60625', '60626', '60629', '60630', '60631', '60632', '60634', '60637', '60638', '60639', '60640', '60641', '60642', '60644', '60645', '60646', '60647', '60651', '60652', '60653', '60654', '60656', '60657', '60659', '60660', '60661', '60664', '60666', '60668', '60670', '60673', '60674', '60675', '60677', '60678', '60680', '60681', '60682', '60684', '60685', '60686', '60687', '60690', '60691', '60693', '60694', '60695', '60697', '60699', '60707', '60804', '60004', '60005', '60006', '60007', '60008', '60015', '60016', '60017', '60018', '60019', '60022', '60025', '60026', '60029', '60038', '60043', '60053', '60055', '60056', '60062', '60065', '60067', '60068', '60070', '60074', '60076', '60077', '60078', '60082', '60089', '60090', '60091', '60093', '60094', '60095', '60159', '60168', '60169', '60172', '60173', '60179', '60192', '60193', '60194', '60195', '60196', '60201', '60202', '60203', '60204', '60208', '60209', '60406', '60409', '60411', '60415', '60419', '60422', '60423', '60425', '60426', '60428', '60429', '60430', '60438', '60439', '60443', '60445', '60452', '60453', '60454', '60455', '60456', '60457', '60459', '60461', '60462', '60464', '60465', '60466', '60467', '60469', '60471', '60472', '60473', '60475', '60476', '60477', '60478', '60480', '60482', '60487', '60521', '60609', '60617', '60619', '60620', '60621', '60628', '60633', '60636', '60643', '60649', '60655', '60706', '60712', '60714', '60803', '60827']

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
        Method to retrieve zip code level data for Cook County, Illinois 
        from the Census Bureau's API.

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
            zip_codes = ','.join(ZIP_CODES)
            query_geo = f"&for={unit}:{zip_codes}&in=state:{STATE}"
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