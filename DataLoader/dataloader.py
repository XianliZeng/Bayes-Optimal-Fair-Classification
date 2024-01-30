
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def arrays_to_tensor(X, Y, Z, XZ, device):
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device), torch.FloatTensor(Z).to(
        device), torch.FloatTensor(XZ).to(device)




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


    all_data = pd.concat([train_data,test_data])
    data_size = len(all_data)
    new_index_all = np.arange(data_size)
    all_data.index = new_index_all


    return all_data


def compas():

    dtypes = [
        ("sex", "category"), ("age", "float32"), ("age_cat", "category"),
        ("race", "category"), ("juv_fel_count", "float32"), ("juv_misd_count", "float32"),
        ("juv_other_count", "float32"), ("priors_count", "float32"), ("c_charge_degree", "category"),
    ]

    data = pd.read_csv("data/compas/compas-scores-two-years.csv")
    data = data[(data['days_b_screening_arrest'] <= 30) &
                (data['days_b_screening_arrest'] >= -30) &
                (data['is_recid'] != -1) &
                (data['c_charge_degree'] != "O") &
                (data['score_text'] != "N/A")]
    data = data[(data['race'] == 'African-American') | (data['race'] == 'Caucasian')]



    data = data[["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                 "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"]]


    categories = ['Male', 'Female']
    cat_dtype = pd.api.types.CategoricalDtype(categories=categories)
    data['sex'] = data['sex'].astype(cat_dtype)
    data['age'] = data['age'].astype('float32')
    categories = ['Less than 25', '25 - 45', 'Greater than 45']
    cat_dtype = pd.api.types.CategoricalDtype(categories=categories)
    data['age_cat'] = data['age_cat'].astype(cat_dtype)

    categories = ['African-American', 'Caucasian']
    cat_dtype = pd.api.types.CategoricalDtype(categories=categories)
    data['race'] = data['race'].astype(cat_dtype)
    data['juv_fel_count'] = data['juv_fel_count'].astype('float32')
    data['juv_misd_count'] = data['juv_misd_count'].astype('float32')
    data['juv_other_count'] = data['juv_other_count'].astype('float32')
    data['priors_count'] = data['priors_count'].astype('float32')
    categories = ['M', 'F']
    cat_dtype = pd.api.types.CategoricalDtype(categories=categories)
    data['c_charge_degree'] = data['c_charge_degree'].astype(cat_dtype)


    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))

    rcode = {
        'Less than 25' : 0,
        '25 - 45': 1,
        'Greater than 45': 2,
    }


    rcode1 = {
        'M' : 0,
        'F': 1,
    }


    rcode2 = {
        'African-American' : 0,
        'Caucasian': 1,
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "age_cat":
                data[k] = np.array([rcode[v.strip()] for v in data[k]])
            elif k == 'c_charge_degree':
                data[k] = np.array([rcode1[v.strip()] for v in data[k]])
            elif k == 'race':
                data[k] = np.array([rcode2[v.strip()] for v in data[k]])
            else:
                data[k] = data[k].cat.codes


    data_size = len(data)
    new_index_all = np.arange(data_size)
    data.index = new_index_all
    return data

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
        alldata = adult('./data/adult/')
        training_set,testing_set =  train_test_split(alldata, test_size=0.3, random_state=seed)
        training_set.index = np.arange(len(training_set))
        testing_set.index = np.arange(len(testing_set))

        self.Z_train_ = training_set['Sex']
        self.Y_train_ = training_set['Target']
        self.X_train_ = training_set.drop(['Target','Sex'],axis=1)


        self.Z_test_ = testing_set['Sex']
        self.Y_test_ = testing_set['Target']
        self.X_test_ = testing_set.drop(['Target','Sex'],axis=1)

        self.X_train_ = pd.get_dummies(self.X_train_)
        self.X_test_ = pd.get_dummies(self.X_test_)

        le = LabelEncoder()
        self.Y_train_ = le.fit_transform(self.Y_train_)
        self.Y_train_ = pd.Series(self.Y_train_, name='>50k')
        self.Y_test_ = le.fit_transform(self.Y_test_)
        self.Y_test_ = pd.Series(self.Y_test_, name='>50k')



    def get_compas_data(self,seed):
        alldata = compas()
        train_val_set, testing_set = train_test_split(alldata, test_size=0.2, random_state=seed)
        training_set, validation_set = train_test_split(train_val_set, test_size=0.2, random_state=seed)
        training_set.index = np.arange(len(training_set))
        validation_set.index = np.arange(len(validation_set))
        testing_set.index = np.arange(len(testing_set))

        self.Z_train_ = training_set['race']
        self.Z_val_ = validation_set['race']
        self.Z_test_ = testing_set['race']



        self.Y_train_ = training_set['two_year_recid']
        self.Y_val_ = validation_set['two_year_recid']
        self.Y_test_ = testing_set['two_year_recid']

        self.X_train_ = training_set.drop(labels=['race','two_year_recid'], axis=1)
        self.X_train_ = pd.get_dummies(self.X_train_)
        self.X_val_ = validation_set.drop(labels=['race','two_year_recid'], axis=1)
        self.X_val_ = pd.get_dummies(self.X_val_)
        self.X_test_ = testing_set.drop(labels=['race','two_year_recid'], axis=1)
        self.X_test_ = pd.get_dummies(self.X_test_)

        le = LabelEncoder()
        self.Y_train_ = le.fit_transform(self.Y_train_)
        self.Y_train_ = pd.Series(self.Y_train_, name='two_year_recid')
        self.Y_val_ = le.fit_transform(self.Y_val_)
        self.Y_val_ = pd.Series(self.Y_val_, name='two_year_recid')
        self.Y_test_ = le.fit_transform(self.Y_test_)
        self.Y_test_ = pd.Series(self.Y_test_, name='two_year_recid')





    def get_lawschool_data(self,seed):
        alldata = lawschool()
        training_set, testing_set = train_test_split(alldata, test_size=0.3, random_state=seed)
        training_set.index = np.arange(len(training_set))
        testing_set.index = np.arange(len(testing_set))


        self.Y_train_ = training_set['admit']
        self.Y_test_ = testing_set['admit']
        self.Z_train_ = training_set['White']
        self.Z_test_ = testing_set['White']
        self.X_train_ = training_set.drop(['White','admit'],axis=1)
        self.X_test_ = testing_set.drop(['White','admit'],axis=1)

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

    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train), \
               (self.X_test, self.Y_test, self.Z_test, self.XZ_test)

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_), \
               (X_test_, Y_test_, Z_test_, XZ_test_)