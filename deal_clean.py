import pandas as pd
import sklearn
file_name = 'cleaned_data.csv'
df = pd.read_csv(file_name)
df['discharge_location'].value_counts()
projection_map = {
    'home': 'Home',
    'unknown': 'Unknown',
    'home health care': 'Home with Services',
    'skilled nursing facility': 'Facility-Based Care',
    'died': 'Deceased / Hospice',
    'rehab': 'Facility-Based Care',
    'hospice': 'Deceased / Hospice',
    'chronic/long term acute care': 'Facility-Based Care',
    'acute hospital': 'Unknown',
    'against advice': 'Unknown',
    'psych facility': 'Facility-Based Care',
    'other facility': 'Facility-Based Care',
    'assisted living': 'Home with Services',
    'healthcare facility': 'Facility-Based Care'
}

# Create a new column with the projected labels
df['discharge_category'] = df['discharge_location'].map(projection_map)
df = df.loc[df['discharge_category'].str.contains('Unknown') == False]
df['length_of_stay'] = (pd.to_datetime(df['discharge_time']) - pd.to_datetime(df['admit_time']))
df['length_of_stay'] = df['length_of_stay'].dt.days
df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.1
                                                             , random_state=42, stratify=df['discharge_category'])
df_test['discharge_category'].value_counts()
df_train['discharge_category'].value_counts()
train_labels = df_train['discharge_category']
train_labels.columns = ['discharge_location']
df_train.drop(columns=['discharge_location', 'discharge_category'], inplace=True)
test_labels = df_test['discharge_category']
test_labels.columns = ['discharge_location']
df_test.drop(columns=['discharge_location', 'discharge_category', 'discharge_time'], inplace=True)
import os
data_dir = 'data_processed'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
df_train.to_csv(os.path.join(data_dir, 'X_train.csv'), index=False)
df_test.to_csv(os.path.join(data_dir, 'X_test.csv'), index=False)
train_labels.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
test_labels.to_csv(os.path.join(data_dir, 'y_test.csv'), index=False)