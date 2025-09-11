import os
import pandas as pd
import json
data_root = './com_data'
data_target = './data_processed'
target_test_ratio = 0.15
if not os.path.exists(data_target):
    os.makedirs(data_target)
X_train = pd.read_csv(os.path.join(data_root, 'X_train.csv'))   
X_test = pd.read_csv(os.path.join(data_root, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(data_root, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(data_root, 'y_test.csv'))
with open(os.path.join(data_root, 'train_profile_concurrent.jsonl'), 'r') as f:
    train_profiles = [json.loads(line) for line in f]
with open(os.path.join(data_root, 'test_profile_concurrent.jsonl'), 'r') as f:
    test_profiles = [json.loads(line) for line in f]
    
len_train = len(X_train)
len_test = len(X_test)

if len_test / (len_train + len_test) < target_test_ratio:
    # need to move some from train to test
    n_total = len_train + len_test
    n_test_target = int(n_total * target_test_ratio)
    n_to_move = n_test_target - len_test
    if n_to_move > 0:
        X_move = X_train.iloc[-n_to_move:]
        y_move = y_train.iloc[-n_to_move:]
        profiles_move = train_profiles[-n_to_move:]
        
        X_train = X_train.iloc[:-n_to_move]
        y_train = y_train.iloc[:-n_to_move]
        train_profiles = train_profiles[:-n_to_move]
        
        X_test = pd.concat([X_test, X_move], ignore_index=True)
        y_test = pd.concat([y_test, y_move], ignore_index=True)
        test_profiles.extend(profiles_move)
        print(f'Moved {n_to_move} samples from train to test.')
elif len_test / (len_train + len_test) > target_test_ratio:
    # need to move some from test to train
    n_total = len_train + len_test
    n_test_target = int(n_total * target_test_ratio)
    n_to_move = len_test - n_test_target
    if n_to_move > 0:
        X_move = X_test.iloc[-n_to_move:]
        y_move = y_test.iloc[-n_to_move:]
        profiles_move = test_profiles[-n_to_move:]
        
        X_test = X_test.iloc[:-n_to_move]
        y_test = y_test.iloc[:-n_to_move]
        test_profiles = test_profiles[:-n_to_move]
        
        X_train = pd.concat([X_train, X_move], ignore_index=True)
        y_train = pd.concat([y_train, y_move], ignore_index=True)
        train_profiles.extend(profiles_move)
        print(f'Moved {n_to_move} samples from test to train.')
        
print(f'Final split: {len(X_train)} train, {len(X_test)} test.')
X_train.to_csv(os.path.join(data_target, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(data_target, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(data_target, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(data_target, 'y_test.csv'), index=False)
train_profiles_re = [{'idx': idx, 'profile': train_profiles[idx]['profile']} for idx in range(len(train_profiles))]
test_profiles_re = [{'idx': idx, 'profile': test_profiles[idx]['profile']} for idx in range(len(test_profiles))]
with open(os.path.join(data_target, 'train_profile_concurrent.jsonl'), 'w') as f:
    for profile in train_profiles_re:
        f.write(json.dumps(profile) + '\n')
with open(os.path.join(data_target, 'test_profile_concurrent.jsonl'), 'w') as f:
    for profile in test_profiles_re:
        f.write(json.dumps(profile) + '\n')

