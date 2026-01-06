from dataset import dataprocess
datapath = './data/'
data_path_train = './data/train.csv'
data_path_test = './data/test.csv'
data_path_val = './data/valid.csv'

print("prepare train data...")
train_dataset = dataprocess(data_path_train,datapath+'train_dataset.pkl')
print("Prepare test data...")
test_dataset = dataprocess(data_path_test,datapath+'test_dataset.pkl')
print("Prepare valid data...")
validate_dataset = dataprocess(data_path_val,datapath+'valid_dataset.pkl')