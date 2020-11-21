import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from sklearn import metrics
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.base import BaseEstimator, TransformerMixin
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor    

class SmilestoDescriptors(BaseEstimator, TransformerMixin):
    """Convert Smiles to Mols"""

    def __init__(self, mode="rdkit") -> str:
        
        #Choose rdkit or ecfp4
        if mode == "rdkit":
            print("Generating rdkit descriptors...")
            self.mode = True
        else:
            print("Generating ecfp4 descriptors...")
            self.mode = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "SmilestoDescriptors":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe: Smiles"""
        
        if not isinstance(X, list):
            X = [X]
        else:
            X = X
            
        #Convert mol to 
        X = X.copy()
        X = [Chem.MolFromSmiles(smile) for smile in X ]   
        
        if self.mode == True:
            
            #Utilize all the rdkit descriptors
            descriptor_names = []
            for descriptor_information in Descriptors.descList:
                descriptor_names.append(descriptor_information[0])
                    
                descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                
            X = [descriptor_calculator.CalcDescriptors(mol) for mol in X ] 
            X = pd.DataFrame(X, columns=descriptor_names)
            
        else:
            descriptors = []
            for mol in X:
                
                #ECFP paramters:
                bi = {}
                fp_radius=2
                fp_length=2048
                
                fp_string = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_length, bitInfo=bi).ToBitString()
                descriptors.append(np.array(list(fp_string), dtype=int))
            
            X = pd.DataFrame(descriptors)
            
        return X
    
#The "transform" part of the code was taken from the following stackoverflow post:
#https://etav.github.io/python/vif_factor_python.html
class DropChollinearityVif(BaseEstimator, TransformerMixin):
    """Drop Chollinear parameters """

    def __init__(self, threshold=10) -> int:
        #Set threshold for ivf parameters: recommendation is 10-5
        #The definition of ivf = 1 - (1/r)
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "DropChollinearityVif":
        """Fit statement to accomodate the sklearn pipeline"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe: Smiles"""
        variables = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
                   for ix in range(X.iloc[:, variables].shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > self.threshold:
                print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                      '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True

        print('Remaining variables:')
        print(X.columns[variables])
        return X.iloc[:, variables]