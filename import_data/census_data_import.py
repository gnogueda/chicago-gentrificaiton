from hashlib import new
from census_data_collector import CensusQuery
import pandas as pd
import numpy as np
import time


def process_data():
    # Retrieve data from Census Bureau
    df_cols = ['zip code tabulation area', 'year', 'med_income', 'med_home_val',
                'med_rent', 'perc_white', 'med_age', 'perc_employed']
    output_dataframe = pd.DataFrame(columns=df_cols)

    for year in range(2012, 2020):
        income_query = CensusQuery()

        year_df = income_query.retrieve_data(
                    year, 'acs/acs5', ['B19013_001E', 'B15003_017E', 'B15003_001E', 
                    'B25077_001E', 'B25064_001E', 'C02003_003E', 'C02003_001E', 
                    'B01002_001E', 'B23025_004E', 'B23025_002E'], 
                    'zip%20code%20tabulation%20area', 'prelim_model')
        if year_df is None:
            year_df = income_query.retry_retrieval()
        
        year_df = year_df.astype('float')

        # Calculate percent high school graduate
        year_df['perc_hs_grad'] = (year_df['B15003_017E'] /
                                        year_df['B15003_001E'])
        year_df = year_df.drop(['B15003_017E', 'B15003_001E'], axis=1)

        # Calculate percent white
        year_df['perc_white'] = (year_df['C02003_003E'] /
                                        year_df['C02003_001E'])
        year_df = year_df.drop(['C02003_003E', 'C02003_001E'], axis=1)

        # Calculate percent employed (of the labor force)
        year_df['perc_employed'] = (year_df['B23025_004E'] /
                                        year_df['B23025_002E'])
        year_df = year_df.drop(['B23025_004E', 'B23025_002E'], axis=1)        
        
        year_df = year_df.rename(columns={'B19013_001E': 'med_income', 
                                    'B25077_001E': 'med_home_val',
                                    'B25064_001E': 'med_rent',
                                    'B01002_001E': 'med_age'})

        year_df['year'] = year
        output_dataframe = (pd.concat([output_dataframe, year_df], ignore_index=True))
        time.sleep(2)

    # Clean data (temporary fix for some data input errors I found)
    output_dataframe.replace(0, method='ffill', inplace=True)
    output_dataframe.replace(-666666666, method='ffill', inplace=True)

    # Create new annual change columns
    new_cols = ['med_income_change', 'med_home_val_change', 'med_rent_change', 
                    'med_age_change', 'perc_white_change', 'perc_hs_grad_change',
                    'perc_employed_change']

    for new_col in new_cols:
        output_dataframe[new_col] = 0
    
    # Iterate over each variable
    for new_col in new_cols:
        num_zip_codes = output_dataframe['zip code tabulation area'].nunique()
        new_vals = np.zeros(num_zip_codes)

        # Calculate change from year prior 
        for year in range(2013, 2020):
            year_prior = year - 1
            old_col = new_col[:-7]
            new = output_dataframe[output_dataframe['year'] == year][old_col]
            old = output_dataframe[output_dataframe['year'] == year_prior][old_col]

            # If variable was percentage, calculate difference, otherwise calc rate of change
            if old_col[:4] == 'perc':
                year_vals = new.to_numpy() - old.to_numpy()
            else:
                year_vals = (new.to_numpy() - old.to_numpy()) / old.to_numpy()

            new_vals = np.append(new_vals, year_vals)
        
        output_dataframe[new_col] = new_vals
            
    output_dataframe.to_csv('output_dataset.csv', mode='x', index=False)
    


if __name__ == "__main__":
    process_data()