
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler





def adult(data_root, display=False):
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_train_data = pd.read_csv(
        data_root + 'adult.data',
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    raw_test_data = pd.read_csv(
        data_root + 'adult.test',
        skiprows=1,
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )


    train_data = raw_train_data.drop(["Education",'fnlwgt'], axis=1)  # redundant with Education-Num
    test_data = raw_test_data.drop(["Education",'fnlwgt'], axis=1)  # redundant with Education-Num
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))

    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                train_data[k] = np.array([rcode[v.strip()] for v in train_data[k]])
                test_data[k] = np.array([rcode[v.strip()] for v in test_data[k]])
            else:
                train_data[k] = train_data[k].cat.codes
                test_data[k] = test_data[k].cat.codes


    train_data["Target"] = train_data["Target"] == " >50K"
    test_data["Target"] = test_data["Target"] == " >50K."

    # 拆分数据框
    all_data = pd.concat([train_data,test_data])
    data_size = len(all_data)
    new_index_all = np.arange(data_size)
    all_data.index = new_index_all


    return all_data



def lawschool():
    rawdata = pd.read_sas('./data/lawschool/lawschs1_1.sas7bdat')
    rawdata = rawdata.drop(['college', 'Year', 'URM', 'enroll'], axis=1)
    rawdata = rawdata.dropna(axis=0)

    data = rawdata[['LSAT', 'GPA', 'Gender', 'resident','White','admit']]
    return data


class CustomDataset():
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y, z = self.X[index], self.Y[index], self.Z[index]
        return x, y, z


class FairnessDataset():
    def __init__(self, dataset,seed, device=torch.device('cuda')):
        self.dataset = dataset
        self.device = device

        if self.dataset == 'AdultCensus':
            self.get_adult_data(seed)
        elif self.dataset == 'COMPAS':
            self.get_compas_data(seed)
        elif self.dataset == 'Lawschool':
            self.get_lawschool_data(seed)
        else:
            raise ValueError('Your argument {} for dataset name is invalid.'.format(self.dataset))
        self.prepare_ndarray()

    def get_adult_data(self,seed):
        alldata = pd.read_csv('data/adult/AdultCensus')
        alldata = alldata.drop('Unnamed: 0',axis = 1)

        training_set, testing_set = train_test_split(alldata, test_size=0.3, random_state=seed)

        self.train_index = training_set.index
        self.train_weights = training_set['weight']
        self.train_neighbours = training_set['neighbour']

        self.Z_train_ = training_set['Sensitive']
        self.Y_train_ = training_set['Label']
        self.keypd = training_set.drop(['Label', 'weight', 'neighbour'], axis=1)

        self.X_train_ = self.keypd.drop(['Sensitive'], axis=1)

        self.keys = self.keypd.keys()

        self.Z_test_ = testing_set['Sensitive']
        self.Y_test_ = testing_set['Label']
        self.X_test_ = testing_set.drop(['Label', 'Sensitive', 'weight', 'neighbour'], axis=1)

        self.X_train_ = pd.get_dummies(self.X_train_)
        self.X_test_ = pd.get_dummies(self.X_test_)

        le = LabelEncoder()
        self.Y_train_ = le.fit_transform(self.Y_train_)
        self.Y_train_ = pd.Series(self.Y_train_, name='>50k')
        self.Y_test_ = le.fit_transform(self.Y_test_)
        self.Y_test_ = pd.Series(self.Y_test_, name='>50k')
    def get_compas_data(self,seed):
        alldata = pd.read_csv('data/compas/COMPAS')
        alldata = alldata.drop('Unnamed: 0',axis = 1)

        training_set, testing_set = train_test_split(alldata, test_size=0.3, random_state=seed)

        self.train_index = training_set.index
        self.train_weights = training_set['weight']
        self.train_neighbours = training_set['neighbour']

        self.Z_train_ = training_set['Sensitive']
        self.Y_train_ = training_set['Label']
        self.keypd = training_set.drop(['Label', 'weight', 'neighbour'], axis=1)

        self.X_train_ = self.keypd.drop(['Sensitive'], axis=1)

        self.keys = self.keypd.keys()

        self.Z_test_ = testing_set['Sensitive']
        self.Y_test_ = testing_set['Label']
        self.X_test_ = testing_set.drop(['Label', 'Sensitive', 'weight', 'neighbour'], axis=1)

        self.X_train_ = pd.get_dummies(self.X_train_)
        self.X_test_ = pd.get_dummies(self.X_test_)

        le = LabelEncoder()
        self.Y_train_ = le.fit_transform(self.Y_train_)
        self.Y_train_ = pd.Series(self.Y_train_, name='two_year_recid')
        self.Y_test_ = le.fit_transform(self.Y_test_)
        self.Y_test_ = pd.Series(self.Y_test_, name='two_year_recid')


    def get_lawschool_data(self,seed):
        alldata = pd.read_csv('data/lawschool/Lawschool')
        alldata = alldata.drop('Unnamed: 0',axis = 1)
        training_set,testing_set =  train_test_split(alldata, test_size=0.3, random_state=seed)

        self.train_index = training_set.index
        self.train_weights = training_set['weight']
        self.train_neighbours = training_set['neighbour']

        self.Z_train_ = training_set['Sensitive']
        self.Y_train_ = training_set['Label']
        self.keypd = training_set.drop(['Label','weight','neighbour'],axis=1)

        self.X_train_ = self.keypd.drop(['Sensitive'],axis=1)

        self.keys = self.keypd.keys()

        self.Z_test_ = testing_set['Sensitive']
        self.Y_test_ = testing_set['Label']
        self.X_test_ = testing_set.drop(['Label','Sensitive','weight','neighbour'],axis=1)


    def prepare_ndarray(self):
        self.normalized = False
        self.X_train = self.X_train_.to_numpy(dtype=np.float64)
        self.Y_train = self.Y_train_.to_numpy(dtype=np.float64)
        self.Z_train = self.Z_train_.to_numpy(dtype=np.float64)
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1, 1)], axis=1)

        self.X_test = self.X_test_.to_numpy(dtype=np.float64)
        self.Y_test = self.Y_test_.to_numpy(dtype=np.float64)
        self.Z_test = self.Z_test_.to_numpy(dtype=np.float64)
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1, 1)], axis=1)

        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def normalize(self):
        self.normalized = True
        scaler_XZ = StandardScaler()
        self.XZ_train = scaler_XZ.fit_transform(self.XZ_train)
        self.XZ_test = scaler_XZ.transform(self.XZ_test)

        scaler_X = StandardScaler()
        self.X_train = scaler_X.fit_transform(self.X_train)
        self.X_test = scaler_X.transform(self.X_test)
        return None

    def get_dataset_in_dataframe(self):
        data_train = pd.DataFrame(self.XZ_train, columns=self.keys)
        data_train.index = self.train_index
        data_train['weight'] = self.train_weights
        data_train['neighbour'] = self.train_neighbours
        data_train['Sen1'] = self.Z_train
        data_train['Label'] = self.Y_train



        data_test =pd.DataFrame(self.XZ_test, columns=self.keys)
        data_test['Sen1'] = self.Z_test
        data_test['Label'] = self.Y_test
        return (data_train,data_test)


