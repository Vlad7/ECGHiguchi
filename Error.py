# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1235VNr5NCxZPdtZI1SbsxRZ7uRTsP4Jm
"""

import numpy as np

def interval_t(size,num_val=100):

    k_stop = size // 2

    k = np.logspace(start=np.log2(2),stop=np.log2(k_stop), base=2,num=num_val,dtype=float)
    
    print(k)

    ki = k.astype(dtype=int, copy=False)

    print(ki)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OWN CODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Because bug in np.logspace with dtype = np.float?
    # 218 - 108 integer; 220 - 110 integer;
    
    return None;


print(interval_t(21))
print(interval_t(220))