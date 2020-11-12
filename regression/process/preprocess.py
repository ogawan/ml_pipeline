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
from regression_model.processing.errors import InvalidModelInputError

class SmilestoDescriptors(BaseEstimator, TransformerMixin):
    """Convert Smiles to Mols"""

    def __init__(self, mode="rdkit") -> str:
        
        #Choose rdkit or ecfp4
        if mode == "rdkit":
            self.mode = True
        else:
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