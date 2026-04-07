# Purpose: Define modified Bernstein and Du Stebbins Model Methods for implementation.
# Author: Chloe Curry

import pandas as pd

# Define Model Constants
ds_k = {'B2O3':1,
'Al2O3':4,
'P2O5':0.7,
'Fe2O3':0,
'TiO2': 0,
'ZrO2':	3,
'HfO2':	0,
'Cr2O3':1,
'TeO2':	0,
'SnO2':	0,
'Sb2O3':1,
'SO3': 0,
'UO3': 1}

ds_r = {'Li2O':	1,
'K2O':	1,
'Cs2O':	1,
'Rb2O':	1,
'CaO':	0.5,
'MgO':	0.5,
'SrO':	0.5,
'BaO':	0.5,
'ZnO':	0.5,
'PbO':	0.5,
'CoO':	0.5,
'La2O3':0.333333333,
'Y2O3': 0.333333333,
'Bi2O3':0.333333333
}

ds_rmax_cons = 0.590
ds_rmax_slope = 0.019

ds_rd1_cons = 0.096
ds_rd1_slope = 0.502

bern_k = {'B2O3':1,
'Al2O3':4,
'P2O5':1,
'Fe2O3':0,
'TiO2': 0,
'ZrO2':	2,
'HfO2':	0,
'Cr2O3':0,
'TeO2':	0,
'SnO2':	0,
'Sb2O3':0,
'SO3': 0,
'UO3': 0.2}

bern_r = {'Li2O':1,
'K2O':	1,
'Cs2O':	1,
'Rb2O':	1,
'CaO':	0.5,
'MgO':	0.5,
'SrO':	0.5,
'BaO':	0.5,
'ZnO':	0.5,
'PbO':	0.5,
'CoO':	0.5,
'La2O3':0.333333333,
'Y2O3': 0.333333333,
'Bi2O3':0.333333333
}

bern_rmax_cons = 0.43
bern_rmax_slope = 0.06

bern_a = 0.24
bern_b = 0.26
bern_c=1.31
bern_d=3.68
bern_e=1.01

# Model Methods 
def calculate_ds(row):
    a = 0
    b = 0
    for oxide, composition in row.items():
        if oxide in ds_k:
            b += ds_k[oxide] * composition
        if oxide in ds_r:
            a += ds_r[oxide] * composition
    r_sum = (row['Na2O'] + a)/(b)
    k_sum = row['SiO2']/(b)
    
    ds_rmax = ds_rmax_slope*k_sum + ds_rmax_cons 
    ds_rd1 = ds_rd1_slope*k_sum + ds_rd1_cons

    if r_sum <= ds_rmax:
        ds_value = r_sum
    elif ds_rmax <= r_sum and r_sum <= ds_rd1:
        ds_value = ds_rmax
    else:
        ds_value = ds_rmax - (r_sum-ds_rd1)*(8+k_sum)/(12*(2+k_sum))

    return ds_value

def calculate_bernstein(row):
    a = 0
    b = 0
    for oxide, composition in row.items():
        if oxide in bern_k:
            b += bern_k[oxide] * composition
        if oxide in bern_r:
            a += bern_r[oxide] * composition

    r_sum = (row['Na2O'] + a)/(b)
    k_sum = row['SiO2']/b

    bern_rmax = bern_rmax_slope*k_sum + bern_rmax_cons

    if r_sum <bern_rmax:
        bern_value = r_sum
    else:
        bern_value = bern_a * (bern_b+r_sum) * (bern_c - (r_sum/(bern_d+bern_e*k_sum)))**5

    return bern_value