from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def big_data_things(filepath = 'dinos.arff'):

    # Load the ARFF file
    data, meta = arff.loadarff(filepath)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    # If you see byte strings, decode them like this:
    # df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # discover some propoerties about the data set
    # print("The data set shape is: ",df.shape)
    # print("Example Rows Looks like: " ,df.head())
    # print(df.isna().sum().values)
    # print(df['class'].nunique())
    # print(df["class"].value_counts(normalize = True))

    # standard scalar all the columns
    scaler = StandardScaler() 

    # first shift all the class labels by 1 so that they are indexed from 0
    df['class'] = df['class'].astype(int)
    # df['class_new'] = df['class'] - 1
    # df.drop('class', axis = 1, inplace = True)
    # we need to create our X and y matrix 
    train_columns = [col for col in df.columns if "class" != col]
    # print(len(train_columns))
    # print(train_columns)
    X = df.loc[:,train_columns]
    y = df["class"]

    # print("Checking dims of X", X.shape)
    # print("Chcking dims of y", y.shape)

    # then we do the train/test split 
    # make an intial train test split but throw away half the data since it is so big
    X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.7, stratify=y)
    # we do it first for test 
    X_temp2, X_test, y_temp2, y_test  = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp)

    X_train, X_val, y_train, y_val = train_test_split(X_temp2, y_temp2, test_size=0.1, stratify=y_temp2)

    # fit the standard scalar on everything
    X_train_scaled = scaler.fit_transform(X_train)
    # apply the same to the val and the test 
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # pack everything into the tensors 
    train_X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    test_X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_y_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    val_X_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    val_y_tensor = torch.tensor(y_val.values, dtype=torch.float)

    # pack this into data sets 
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
    test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
    val_dataset = TensorDataset(val_X_tensor, val_y_tensor)

    # save this shit 
    torch.save(train_dataset, f"train_dataset.pt")
    torch.save(test_dataset, f"test_dataset.pt")
    torch.save(val_dataset, f"val_dataset.pt")




big_data_things(filepath = 'dinos.arff')