import numpy as np

def read_in_data(cancer_code, use_small, use_beta=False, root_path='', outlier = False):
    save_path = root_path + "data_preprocessing/dataset/"
    
    if use_beta:
        folder = 'beta_values'
    else:
        folder = 'm_values'
        
    if cancer_code == '':
        cancer_code = 'all'
        
    if use_small:
        cancer_code = cancer_code + '_small'
        
    if outlier:
        panda = 'outliers/'
    else:
        panda = 'pandas/'
    
    print("Getting values and diagnoses from: ")
    print(save_path + panda+folder+'/TCGA-'+cancer_code + '.csv')
    print(save_path + panda + 'diagnoses/TCGA-'+cancer_code + '.csv')
    
    import pandas
    values = pandas.read_csv(save_path + panda +folder+'/TCGA-'+cancer_code + '.csv', sep='\t', index_col=0)
#     if cancer_code == 'all':
#         values = values.drop("probe", axis = 1) # for the concatenated data, we need to remove the 
    values = np.matrix(values)
    
    diagnoses = pandas.read_csv(save_path + panda + 'diagnoses/TCGA-'+cancer_code + '.csv', sep='\t', index_col = 0)
    diagnoses = np.matrix(diagnoses)
    diagnoses = np.ravel(diagnoses)
    
    print("m_value and diagnoses shapes:")
    print(values.shape)
    print(diagnoses.shape)
    return values, diagnoses


# given m values, removes Inf and -Inf which are caused by a beta value of 0
# TODO: change the data processing code to deal with this instead
def deal_with_Inf(m_values_mat):
    # I will for now set -Inf to the min value and Inf to the max value
    # I did want to set it to the min and max possible value, but this causes overflow errors
    max_val = np.max(m_values_mat[np.logical_not(np.isinf(m_values_mat))]) # looking at m values that arn't Inf
    min_val = np.min(m_values_mat[np.logical_not(np.isinf(m_values_mat))]) 
    
    indices = np.argwhere(np.isinf(m_values_mat)) # where m values is +/- Inf
    for i in range(len(indices)):
        if m_values_mat[indices[i][0], indices[i][1]] > 0:
            m_values_mat[indices[i][0], indices[i][1]] = max_val
        else:
            m_values_mat[indices[i][0], indices[i][1]] = min_val
    return m_values_mat

# given m values and their classes, removes samples so that there is an equal number of samples in each class
# currently only works for 2 classes, with 0, 1 diagnoses
def balance_classes(m_values, diagnoses):
    class_nums = [np.sum(diagnoses == 0), np.sum(diagnoses == 1)]
    lowest_class = np.argmin(class_nums)
    num_in_lowest = np.sum(diagnoses == lowest_class)
    highest_class = np.argmax(class_nums)
    # so we need to randomly sample lowest_class elements from the highest class
    possible_indices = np.where(diagnoses == highest_class)[0]
    sampled_indices = np.random.choice(possible_indices, num_in_lowest, replace = False)
    lowest_indices = np.where(diagnoses == lowest_class)[0]
    indices_to_take = np.append(sampled_indices, lowest_indices)
    indices_to_take.sort() # sort them so retaining the order in m_values and diagnoses
    return diagnoses[indices_to_take], m_values[:, indices_to_take]
    

def get_train_and_test(cancer_code, use_small, remove_inf = True, use_beta = False, root_path = '', model_path = '', model_type = '', balanced_classes = False, seed = None):
    # read in the data:
    m_values, diagnoses = read_in_data(cancer_code, use_small, use_beta, root_path)
    
    if balanced_classes:
        diagnoses, m_values = balance_classes(m_values, diagnoses)
    
    # put probes into columns and samples into rows (train_test_split needs this)
    m_values = m_values.transpose()

    # split up into train and test
    from sklearn.model_selection import train_test_split
    if seed == None:
        m_values_train, m_values_test, diagnoses_train, diagnoses_test = train_test_split(m_values, diagnoses, test_size = 0.25, stratify = diagnoses)
    else:
        m_values_train, m_values_test, diagnoses_train, diagnoses_test = train_test_split(m_values, diagnoses, test_size = 0.25, stratify = diagnoses, random_state = seed)
    
    print("m values train, m values test, diagnoses train, diagnoses test shapes:")
    print(m_values_train.shape, m_values_test.shape, diagnoses_train.shape, diagnoses_test.shape)
    
    test_indices = [np.argwhere(np.all(m_values == i, axis = 1))[0][0] for i in m_values_test] # assumes all m value matrices are unique (highly highly unlikely two samples have the same m values)
    
    
    # verifying this is correct:
#     print("test indices:")
#     print(test_indices)
#     print(len(test_indices))
#     print("should be test diagnoses:")
#     print(diagnoses[test_indices])
#     print(diagnoses_test)
#     print("should be test m values:")
#     print(m_values[test_indices])
#     print(m_values_test)
    
    np.savetxt(fname = model_path + "test_data_indices_"+model_type, X = test_indices, fmt = "%d")
    
    
    # remove Inf values:
    if remove_inf:
        m_values_train = deal_with_Inf(m_values_train)
        m_values_test = deal_with_Inf(m_values_test)

    return m_values_train, m_values_test, diagnoses_train, diagnoses_test

    
# all the external data is processed for the specific cancer type, not the general multiclass model. 
# The probes in the multiclass model is a subset of the probes in any cancer type, so here we take out probes to get the probes for the multiclass model.
def get_multiclass_probes(m_values, cancer_type):
    import pandas as pd
    # find the probes currently in use
    try:
        current_probes = pd.read_csv('../../data_preprocessing/dataset/pandas/m_values/TCGA-'+ cancer_type + '.csv', sep  = '\t', usecols = [0])
    except:
        current_probes = pd.read_csv('../data_preprocessing/dataset/pandas/m_values/TCGA-'+ cancer_type + '.csv', sep  = '\t', usecols = [0])
    assert len(current_probes) == m_values.shape[0]
    
    # find the probes we want
    try:
        wanted_probes = pd.read_csv('../../data_preprocessing/dataset/pandas/TCGA-all-probes.csv', sep = '\t', usecols = [1])
    except:
        wanted_probes = pd.read_csv('../data_preprocessing/dataset/pandas/TCGA-all-probes.csv', sep = '\t', usecols = [1])
    
    m_values['probe'] =  list(current_probes['probe'])
    
    m_values = wanted_probes.merge(m_values, how = "left", left_on = "probe", right_on = "probe")
    
    assert len(wanted_probes) == m_values.shape[0]
    return m_values

# use when testing (already processed) external datasets
# has_header indicates whether the m_values.csv includes column names or not
def get_external_data(cancer_type, multiclass, remove_inf, has_header = True, use_raw = False, root_path = '', use_training_imputation = False, use_corrected_PRAD = False):
    if use_raw:
        to_add = '_from_raw'
    else:
        to_add = ''
    if use_corrected_PRAD:
        to_add = to_add + '_corrected'
    if use_training_imputation:
        to_add = to_add + '_training_imputation_constant'
    import pandas as pd
    
    if has_header:
        m_values = pd.read_csv( root_path + 'm_values' + to_add + '.csv', sep = '\t', index_col = 0)
    else:
        m_values = pd.read_csv( root_path + 'm_values' + to_add + '.csv', sep = '\t', header = None)
    print("Using path ", root_path + 'm_values' + to_add + '.csv')
    if use_corrected_PRAD:
        diagnoses = pd.read_csv( root_path + 'diagnoses_corrected.csv', sep = '\t')
        diagnoses = list(diagnoses['x'])
    else:
        diagnoses = pd.read_csv( root_path + 'diagnoses.csv', header = None)
    
    if multiclass:
        m_values = get_multiclass_probes(m_values, cancer_type)
        m_values = m_values.drop('probe', axis = 1)
    
    import numpy as np
    m_values = np.matrix(m_values)
    m_values = m_values.transpose()
    m_values = m_values.astype('float')
    print(m_values.shape)
    
    diagnoses = np.matrix(diagnoses)
    diagnoses = np.ravel(diagnoses)
    print(diagnoses.shape)
    
    # remove Inf values:
    if remove_inf:
        m_values = deal_with_Inf(m_values)
    
    return m_values, diagnoses