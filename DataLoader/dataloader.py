import os
import pickle
import urllib.request

import datasets
import folktables

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

def acsincome():
    sensitive_attr = 'SEX'
    target = 'PINCP'
    features = [
        'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP'
    ]
    categories = {
        "COW": {
            1.0: ("Employee of a private for-profit company or"
                  "business, or of an individual, for wages,"
                  "salary, or commissions"),
            2.0: ("Employee of a private not-for-profit, tax-exempt,"
                  "or charitable organization"),
            3.0:
                "Local government employee (city, county, etc.)",
            4.0:
                "State government employee",
            5.0:
                "Federal government employee",
            6.0: ("Self-employed in own not incorporated business,"
                  "professional practice, or farm"),
            7.0: ("Self-employed in own incorporated business,"
                  "professional practice or farm"),
            8.0:
                "Working without pay in family business or farm",
            9.0:
                "Unemployed and last worked 5 years ago or earlier or never worked",
        },
        "SCHL": {
            1.0: "No schooling completed",
            2.0: "Nursery school, preschool",
            3.0: "Kindergarten",
            4.0: "Grade 1",
            5.0: "Grade 2",
            6.0: "Grade 3",
            7.0: "Grade 4",
            8.0: "Grade 5",
            9.0: "Grade 6",
            10.0: "Grade 7",
            11.0: "Grade 8",
            12.0: "Grade 9",
            13.0: "Grade 10",
            14.0: "Grade 11",
            15.0: "12th grade - no diploma",
            16.0: "Regular high school diploma",
            17.0: "GED or alternative credential",
            18.0: "Some college, but less than 1 year",
            19.0: "1 or more years of college credit, no degree",
            20.0: "Associate's degree",
            21.0: "Bachelor's degree",
            22.0: "Master's degree",
            23.0: "Professional degree beyond a bachelor's degree",
            24.0: "Doctorate degree",
        },
        "MAR": {
            1.0: "Married",
            2.0: "Widowed",
            3.0: "Divorced",
            4.0: "Separated",
            5.0: "Never married or under 15 years old",
        },
        "SEX": {
            1.0: "Male",
            2.0: "Female"
        },
        "RAC1P": {
            1.0: "White alone",
            2.0: "Black or African American alone",
            3.0: "American Indian alone",
            4.0: "Alaska Native alone",
            5.0: ("American Indian and Alaska Native tribes specified;"
                  "or American Indian or Alaska Native,"
                  "not specified and no other"),
            6.0: "Asian alone",
            7.0: "Native Hawaiian and Other Pacific Islander alone",
            8.0: "Some Other Race alone",
            9.0: "Two or More Races",
        },
    }

    # load raw ACS data
    get_data_fn = lambda: folktables.ACSDataSource(
        survey_year='2018',
        horizon='1-Year',
        survey='person',
    ).get_data(download=True)
    raw = cache_dataset(f"./data/acsincome/raw_dataset.pkl", get_data_fn)
    df = folktables.adult_filter(raw)

    label_transform = lambda x: (x > 50000).astype(int)
    X_df, y_arr, z_arr = folktables.BasicProblem(
        features=features + [sensitive_attr],
        target=target,
        target_transform=label_transform,
        group=sensitive_attr,
        postprocess=lambda x: np.nan_to_num(x, -1),
    ).df_to_pandas(df, categories=categories, dummies=True)

    y = pd.Series(y_arr.squeeze(), name=target)
    z = pd.Series((z_arr.squeeze() - 1).astype(int), name=sensitive_attr)
    data = pd.concat([X_df, y, z], axis=1)
    #data.to_csv("data/acsincome/acsincome")
    #print("SAVED TO ACSIncome CSV")
    return data

def cache_dataset(path, get_data_fn):
    if os.path.exists(path):
      with open(path, "rb") as f:
        data = pickle.load(f)
    else:
      directory = os.path.dirname(path)
      os.makedirs(directory, exist_ok=True)
      data = get_data_fn()
      with open(path, "wb") as f:
        pickle.dump(data, f)
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
        elif self.dataset == 'ACSIncome':
            self.get_acsincome_data(seed)
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
        
    def get_acsincome_data(self, seed):
        alldata = acsincome()                            
        train_df, test_df = train_test_split(alldata, test_size=0.3, random_state=seed)
        train_df.index = np.arange(len(train_df))
        test_df.index  = np.arange(len(test_df))

        self.Y_train_ = train_df['PINCP']
        self.Z_train_ = train_df['SEX']
        self.Y_test_  = test_df['PINCP']
        self.Z_test_  = test_df['SEX']

        self.X_train_ = train_df.drop(['PINCP', 'SEX'], axis=1)
        self.X_test_  = test_df.drop (['PINCP', 'SEX'], axis=1)

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