import pandas as pd

ChemicalProcessData = pd.read_csv("D:/Statistics (Python)/Cases/Chemical Process Data/ChemicalProcess.csv")

ChemicalProcessData.head(n=10)

dum_ChemicalProcessData = pd.get_dummies(ChemicalProcessData, drop_first=True)

dum_ChemicalProcessData.head(n=10)

from sklearn.impute import KNNImputer

X=pd.read_csv("D:/Statistics (Python)/Cases/Chemical Process Data/ChemicalProcess.csv")

imputer = KNNImputer(n_neighbors=2)
imputer.fit_transform(X)

#SimpleImputer
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
ChemicalProcessDataImputed = imp.fit_transform(dum_ChemicalProcessData)

ChemicalProcessDataImputed = pd.DataFrame(ChemicalProcessDataImputed,columns= dum_ChemicalProcessData.columns)

dum_ChemicalProcessData.shape
ChemicalProcessDataImputed.shape
ChemicalProcessDataImputed.shape

#StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(ChemicalProcessData)
ChemicalProcessDatascaled=scaler.transform(ChemicalProcessData)

# OR
ChemicalProcessDatascaled=scaler.fit_transform(ChemicalProcessData)

np.mean(ChemicalProcessDatascaled[:,0]), np.std(ChemicalProcessDatascaled[:,0])
np.mean(ChemicalProcessDatascaled[:,1]), np.std(ChemicalProcessDatascaled[:,1])
np.mean(ChemicalProcessDatascaled[:,2]), np.std(ChemicalProcessDatascaled[:,2])
np.mean(ChemicalProcessDatascaled[:,3]), np.std(ChemicalProcessDatascaled[:,3])
np.mean(ChemicalProcessDatascaled[:,4]), np.std(ChemicalProcessDatascaled[:,4])

# Converting numpy array to pandas
ChemicalProcessData = pd.DataFrame(ChemicalProcessDatascaled,columns=ChemicalProcessData.columns,
                       index=ChemicalProcessData.index)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax.fit(ChemicalProcessData)
minmaxChemicalProcessData = minmax.transform(ChemicalProcessData)
minmaxChemicalProcessData[1:5,]

# OR
minmaxChemicalProcessData = minmax.fit_transform(ChemicalProcessData)
