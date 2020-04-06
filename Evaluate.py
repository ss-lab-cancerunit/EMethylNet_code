# Functions for evaluating models

from sklearn import metrics

# given a confusion matrix, plots it so it looks nice
def plot_confusion_matrix(conf_mat, model_type, save_path = None, small_text = False, cancer_type = '', font_size = 36):
    import seaborn as sns
    if small_text:
        sns.set(font_scale=1.2)
    else:
        sns.set(font_scale=1.75)
    
#     sns.set(rc={'figure.figsize':(12,12)})
    import matplotlib.pyplot as plt
    plt.clf()
    plt.rcParams.update({'font.size': font_size})
    if small_text == True:
        plt.rcParams.update({'font.size': 12})
        if conf_mat.shape[0] == 13: # for when we get rid of LUAD
            labels = ['Normal', 'BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUSC', 'PRAD', 'THCA', 'UCEC'] # taken from merging_cancer_types.R
        else:
            labels = ['Normal', 'BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA', 'UCEC'] # taken from merging_cancer_types.R
        plot = sns.heatmap(conf_mat, annot=True, fmt='d', square = True, cbar = False, cmap='Blues', xticklabels = labels, yticklabels = labels, linewidths=1, linecolor="black", annot_kws = {"size": 9})
    else: 
        labels = ['Normal', cancer_type]
        plot = sns.heatmap(conf_mat, annot=True, fmt='d', square = True, cbar = False, cmap='Blues', xticklabels = labels, yticklabels = labels, linewidths=2, linecolor="black", annot_kws = {"size": 20})
    plot.set(xlabel='Predicted label', ylabel='True label')
    plt.suptitle(cancer_type, fontsize=20)
    
    fig = plot.get_figure()
    if save_path == None:
        fig.savefig('figs/confusion_matrix_'+model_type+'.svg', bbox_inches='tight')
    else:
        fig.savefig(save_path,  bbox_inches='tight')
    
    
# given the test predictions and ground truths, makes a confusion matrix where the boxes contain the sample's indices rather than counts. This enables us to find if models missclassify the same samples or not.
def make_labelled_conf_mat(diagnoses_test, predictions, num_classes, model_type, save_folder = ''):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 26})

    import numpy as np
    # make matrix of size numclasses x numclasses
    mat = np.full((num_classes, num_classes), '', dtype = object)
    counts = np.full((num_classes, num_classes), 0, dtype = int)
    
    # for each sample, put it in [diagnoses_test[sample], predictions[sample]]
    for i in range(len(diagnoses_test)):
        mat[diagnoses_test[i], predictions[i]] = str(mat[diagnoses_test[i], predictions[i]]) + str(i) + ', '
        counts[diagnoses_test[i], predictions[i]] = counts[diagnoses_test[i], predictions[i]] + 1
        
        if counts[diagnoses_test[i], predictions[i]] % 2 == 0: # manually putting in new lines so the text wraps (setting the annot_kws param below doesn't seem to do anything)
            mat[diagnoses_test[i], predictions[i]] = str(mat[diagnoses_test[i], predictions[i]]) + '\n'
    
    # remove diaganols as these contain too many labels and are not as interesting:
    for i in range(num_classes):
        mat[i,i] = ''
    
    conf_mat = metrics.confusion_matrix(diagnoses_test, predictions)
    import matplotlib.pyplot as plt
    plt.clf()
    import seaborn as sns
    plot = sns.heatmap(conf_mat, annot=mat, fmt = "", square = True, cbar = False, cmap='Blues', linewidths=2, linecolor="black", annot_kws = {"wrap":True, "size": 10})
    plot.set(xlabel='Predicted label', ylabel='True label')
    fig = plot.get_figure()
    fig.savefig(save_folder + 'figs/labelled_confusion_matrix_'+model_type+'.svg')

# small_text is for making the confusion matrix - with many classes, set to True
def print_evaluation(fitted, m_values_test, diagnoses_test, model_type, predictions=None, prob_predictions=[], no_roc = False, cm_labels = [], small_text = False, cancer_type = '', save_folder = ''):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 26})

    if fitted != None: # if a fitted, find predictions (else use predictions)
        predictions = fitted.predict(m_values_test)

    predictions
    diagnoses_test

    (predictions == diagnoses_test)
    
    print(diagnoses_test)
    print(predictions)
    
    accuracy = metrics.accuracy_score(diagnoses_test, predictions)
    if cm_labels == []:
        conf_mat = metrics.confusion_matrix(diagnoses_test, predictions)
    else:
        conf_mat = metrics.confusion_matrix(diagnoses_test, predictions, labels = cm_labels)
    precision = metrics.precision_score(diagnoses_test, predictions, average = None)
    recall = metrics.recall_score(diagnoses_test, predictions, average = None)
    f1 = metrics.f1_score(diagnoses_test, predictions, average = None)
    mcc = metrics.matthews_corrcoef(diagnoses_test, predictions) # matthews correlation coefficient is supposed to be better at dealing with class imbalance than f1
    # NOTE: with multiple classes, mcc does not have a minimum value of -1 (it will be between -1 and 0). The max value is always 1
    roc_auc = None
    
    #  for roc_auc we need a one hot format: (only works if we have fitted or prob_predictions)
    try:
        if fitted != None and no_roc == False:
            import pandas as pd
            diagnoses_one_hot = pd.get_dummies(pd.Series(diagnoses_test))
            predictions_one_hot = fitted.predict_proba(m_values_test)
            roc_auc = metrics.roc_auc_score(diagnoses_one_hot, predictions_one_hot, average = None)
    except:
        print("Couldn't calculate roc_auc")
        
    try:
        if prob_predictions != [] and no_roc == False:
            import pandas as pd
            diagnoses_one_hot = pd.get_dummies(pd.Series(diagnoses_test))        
            roc_auc = metrics.roc_auc_score(diagnoses_one_hot, prob_predictions, average = None)
    except:
        print("Couldn't calculate roc_auc")

    print("Accuracy: ")
    print(accuracy)
    print("Conf mat:")
    print(conf_mat)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Matthews Correlation Coeficient: ", mcc)
    
    if fitted != None or prob_predictions != []:
        print("roc_auc for each class: ", roc_auc)
    plot_confusion_matrix(conf_mat, model_type, small_text = small_text, cancer_type = cancer_type, save_path = save_folder + 'confusion_matrix_'+model_type+'.svg')
    
    import pandas as pd
    model_metrics = pd.DataFrame(data = {'name': ["Accuracy", "Precision", "Recall", "f1", "mcc", "roc_auc"], 'values': [accuracy, precision, recall, f1, mcc, roc_auc]})
    print(model_metrics)
    
    model_metrics.to_csv(save_folder + "metrics_" + model_type + ".csv", index = False)
    
    return predictions




    
from sklearn.metrics import roc_curve, precision_recall_curve

# can plot ROC curves (curve_type='roc') or precision recall curves (curve_type='precision_recall')
def plot_curve(curve_type, diagnoses, probs, num_classes, model_type, cancer_type = '', save_folder = ''):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 30})
    plt.clf()
    
    if curve_type == 'roc':
        stats_per_class = [roc_curve(diagnoses, probs[:,class_num], pos_label = class_num) for class_num in range(num_classes)]
        xs = [stats_per_class[i][0] for i in range(num_classes)]
        ys = [stats_per_class[i][1] for i in range(num_classes)]
    elif curve_type == 'precision_recall':
        stats_per_class = [precision_recall_curve(diagnoses, probs[:, class_num], pos_label = class_num) for class_num in range(num_classes)]
        xs = [stats_per_class[i][1] for i in range(num_classes)] # recall is x
        ys = [stats_per_class[i][0] for i in range(num_classes)] # precision is y
    else:
        print("Curve type ", curve_type, " not known.")
        return 1
    
#     print(xs)
#     print(ys)
    
    # using matplotlib:
#     plt.figure(figsize=(13,9))
#     # from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#     colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green']
#     for i, color in zip(range(num_classes), colors):
#         plt.plot(fprs[i], tprs[i], color=color, lw=2, label='ROC curve of class {0}'.format(i))
#     plt.legend()
#     plt.show()
    
    # using seaborn
    import seaborn as sns
#     sns.set(rc={'figure.figsize':(12,12)})
#     sns.set(style="whitegrid")
#     sns.set(font_scale=1.5)
    
    
    # create dataframe that includes tpr and fpr info for all classes (with each class having its own column):
    import pandas as pd
    data = pd.DataFrame(ys[0], xs[0], columns=[0])
    for i in range(1, num_classes):
        data = data.append(pd.DataFrame(ys[i], xs[i], columns=[i]))
    if num_classes == 14:
        data.columns = ['Normal', 'BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA', 'UCEC'] # taken from merging_cancer_types.R
    elif num_classes == 13: # for when we take out LUAD
        data.columns = ['Normal', 'BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUSC', 'PRAD', 'THCA', 'UCEC'] # taken from merging_cancer_types.R
    elif data.shape[1] == 1:
        data.columns = [cancer_type] # for an all cancer dataset (I think..!)
    else:
        data.columns = ['Normal', cancer_type]
    print(data)
    
    # plotting
    if (num_classes == 14 or num_classes == 13):
        import matplotlib.pyplot as plt
        plt.clf()
        new = sns.color_palette("Paired")
        new.append((0.43,0.43,0.43))
        new.append((0,0,0)) # adding on two more colours to get 14 colours
        new.reverse()
        temp = new[12] # swapping the colours around so a nice colour is on top
        new[12] = new[13]
        new[13] = temp
        if num_classes == 13:
            del new[9]
    else:
        colours = ['#d95f02','#7570b3']
        new = sns.color_palette(colours)
    
    if cancer_type == '':
        lw = 1
    else:
        lw = 2
    
    
    sns.set(style="whitegrid")
    sns.set(font_scale=1.5, rc={'axes.facecolor':'white', 'figure.facecolor':'white','grid.color': '.8'}) # for some reason changing the font scale messes up the white background
    
    plot = sns.lineplot(data = data, palette=new, linewidth=lw, legend="full", estimator=None, dashes=False) # no estimator means we can have the same x value for multiple y values (creating vertical lines)
    plt.ylim(0, None) # make sure y axis starts at 0
    if num_classes in [13,14]:
        plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        if curve_type == 'roc':
            plt.legend(title='Class', loc = 'lower right', fontsize = 16)
        else:
            plt.legend(title='Class', loc = 'lower left', fontsize = 16)
            
    if curve_type == 'roc':
#         plt.title('ROC curve for each class')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
    elif curve_type == 'precision_recall':
#         plt.title('Precision recall curve for each class')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    # save
    fig = plot.get_figure()
    fig.savefig(save_folder + curve_type+'_curve_'+model_type+'.svg', bbox_inches = 'tight')

    
# eg. usage for binary xgboost:
# from Evaluate import load_data_from_indices
# Xtest, ytest = load_data_from_indices('../xgboost/test_data_indices_XGBoost_'+cancer_type, cancer_type, True)
# get_pr_auc(cancer_type, '../xgboost/figs/metrics_xgboost_' + cancer_type + '.csv', model_path = '../xgboost/saved_models/xgboost_model_'+cancer_type + '.pkl')
def get_pr_auc(cancer_type, metrics_path, model_path, Xtest, ytest):
    from sklearn import metrics as mets 
    
    # checking if we already caluclated and stored it before:
    import pandas as pd
    metrics = pd.read_csv(metrics_path, index_col=0)
    if 'pr_auc' in metrics.index:
        print(' Can just quote: ', metrics.loc['pr_auc'].values[0].strip('[]').split())
        pr_auc = metrics.loc['pr_auc'].values[0].strip('[]').split()
        if cancer_type == '':
            return([float(p) for p in pr_auc])
        else:
            return(float(pr_auc[0]), float(pr_auc[1]))

    else: # working it out and then saving it
        
        if 'NN' in model_path:
            import keras
            from keras.models import load_model
            model = load_model('../NNs/FoundFeatureNN/FeatureNN__best_model')
        else:
            import joblib
            model = joblib.load(model_path)

        diagnoses_test = ytest
        predictions_one_hot = model.predict_proba(Xtest)
        
        num_classes = predictions_one_hot.shape[1]
        
        stats_per_class = [mets.precision_recall_curve(ytest, predictions_one_hot[:, class_num], pos_label = class_num) for class_num in range(num_classes)]
        xs = [stats_per_class[i][1] for i in range(num_classes)] # recall is x
        ys = [stats_per_class[i][0] for i in range(num_classes)] # precision is y
        
        if cancer_type == '':
            aucs = [mets.auc(xs[i], ys[i]) for i in range(len(xs))]
            
        else:
            auc_0 = mets.auc(xs[0], ys[0])
            auc_1 = mets.auc(xs[1], ys[1])

        # add to metrics file:
        metrics = pd.read_csv(metrics_path, index_col=0)
        if cancer_type == '':
            pr_auc = ' '.join(map(str, aucs))
        else:
            pr_auc = ' '.join([str(auc_0), str(auc_1)])
            
        pr_auc = '['+pr_auc + ']'
        pr_row = pd.DataFrame(['pr_auc', pr_auc]).transpose()
        pr_row.columns = ['name', 'values']
        pr_row = pr_row.set_index('name')
        metrics = metrics.append(pr_row)

        metrics.to_csv(metrics_path, index_label = 'name')
        
        if cancer_type == '':
            return aucs
        else:
            return(auc_0, auc_1)
        # Note that here we use the auc function on the pr curve, rather than using the average_precision metric
        # I decided average precision is too approximate. See https://datascience.stackexchange.com/questions/52130/about-sklearn-metrics-average-precision-score-documentation

    
# currently built for binary xgboost:
# get_roc_auc(cancer_type, '../xgboost/figs/metrics_xgboost_' + cancer_type + '.csv')
def get_roc_auc(cancer_type, metrics_path):
    import pandas as pd
    metrics = pd.read_csv(metrics_path, index_col=0)
    roc_aucs = metrics.loc['roc_auc'].values[0].strip('[]').split()
    if cancer_type == '':
        return [float(r) for r in roc_aucs]
    else: # for binary
        assert roc_aucs[0] == roc_aucs[1] # I expect these to be the same (for binary classification at least)
        return(float(roc_aucs[0]))

# given lists of aucs for different cancer types, plots two bar charts
# the auc_0s and auc_1s are the precision recall curve aucs
def plot_pr_and_roc_auc_bars(cancer_types, auc_0s, auc_1s, roc_aucs, pr_save_path, roc_save_path):
    import seaborn as sb
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    if auc_1s == None and len(cancer_types) == 14:
        plt.rcParams.update({'font.size': 16})
    else:
        plt.rcParams.update({'font.size': 20})
    
    
    if auc_1s == None: # if we just have one lot of pr aucs to plot (in the multiclass case)
        data = pd.DataFrame([cancer_types, auc_0s]).transpose()
        data.columns = ['Cancer type', 'PR AUC']
    else: # for the binary case where we have normal and cancer aucs for each cancer type
        df1 = pd.DataFrame([cancer_types, auc_0s, np.repeat('normal', len(cancer_types))]).transpose()
        df2 = pd.DataFrame([cancer_types, auc_1s, np.repeat('cancer', len(cancer_types))]).transpose()

        data = pd.concat([df1, df2])
        data.columns = ['Cancer type', 'PR AUC', 'Normal or cancer']
    data = data.dropna() # removing na rows
    
    roc_data = pd.DataFrame([cancer_types, roc_aucs]).transpose()
    roc_data.columns = ['Cancer type', 'ROC AUC']
    roc_data = roc_data.dropna() # removing na rows

    if auc_1s == None:
        colours = ['#d95f02' if ct == 'Normal' else '#7570b3' for ct in data['Cancer type']]
        sb.barplot(data = data, y = 'Cancer type', x = 'PR AUC', palette = colours)
    else:
        colours = ['#d95f02','#7570b3']
        new = sb.color_palette(colours)
        sb.barplot(data = data, y = 'Cancer type', x = 'PR AUC', hue = 'Normal or cancer', palette = new)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # make the legend appear on the right
    
    min_data_point = min(min(data['PR AUC']), min(roc_data['ROC AUC']))
    if min_data_point > 0.9:
        xlim_lower = 0.9
    elif min_data_point > 0.7:
        xlim_lower = 0.7
    else:
        xlim_lower = 0
    
    plt.xlim(xlim_lower, 1)
    plt.savefig(pr_save_path, bbox_inches = 'tight')
    plt.figure()

    colours = ['#d95f02' if ct == 'Normal' else '#7570b3' for ct in data['Cancer type']]
    sb.barplot(data = roc_data, y = 'Cancer type', x = 'ROC AUC', palette = colours)
    plt.xlim(xlim_lower, 1)
    plt.savefig(roc_save_path, bbox_inches = 'tight')
    

    
# loads test data from a test indices file
def load_data_from_indices(indices_path, cancer_code, remove_inf):
    import numpy as np
    indices = np.loadtxt(indices_path)
    from get_train_and_test import read_in_data
     
    root_path = '../'
    m_values, diagnoses = read_in_data(cancer_code, False, False, root_path)
    
    m_values = m_values.transpose()
    m_values.shape
    
    if remove_inf:
        from get_train_and_test import deal_with_Inf
        m_values = deal_with_Inf(m_values)
    
    Xtest = m_values[indices.astype(int), :]
    ytest = diagnoses[indices.astype(int)]
    
    return Xtest, ytest
    
    
# given the path of a saved model, loads it and runs evaluation metrics on it
def load_and_eval(model_path, indices_path, save_name, cancer_code='', use_small=False, remove_inf = True, folder_name = 'figs/'):
    # get model
    import joblib
    model = joblib.load(model_path)
    
    # get data
    import numpy as np
    indices = np.loadtxt(indices_path)
    from get_train_and_test import read_in_data
    root_path = '../' # NOTE: you may need to change this. It should be pointing to the folder that contains data_preprocessing.
    m_values, diagnoses = read_in_data(cancer_code, False, False, root_path)
    
    m_values = m_values.transpose()
    m_values.shape
#     Xtrain, Xtest, ytrain, ytest = get_train_and_test(cancer_code=cancer_code, use_small=use_small, remove_inf = False)
    
    if remove_inf:
        from get_train_and_test import deal_with_Inf
        m_values = deal_with_Inf(m_values)
    
    Xtest = m_values[indices.astype(int), :]
    ytest = diagnoses[indices.astype(int)]
    
    confidence = model.predict_proba(Xtest)
    import numpy as np
    num_classes = len(np.unique(ytest, axis=0))
    print("num classes is: ", num_classes)
    
    # evaluate
    if num_classes == 14:
        small_text = True
    else:
        small_text = False
    print_evaluation(model, Xtest, ytest, save_name, cancer_type = cancer_code, save_folder = folder_name, small_text = small_text)
    
    
    predictions = model.predict(Xtest)
#     make_labelled_conf_mat(ytest, predictions, num_classes, save_name)
    
    plot_curve('roc', ytest, confidence, num_classes, save_name, cancer_type = cancer_code, save_folder = folder_name)
    plot_curve('precision_recall', ytest, confidence, num_classes, save_name, cancer_type = cancer_code, save_folder = folder_name)


    
# Example usage of load_and_eval:
# from xgboost directory, run:
# import sys
# sys.path.append('/Tank/methylation-patterns-code/methylation-patterns-izzy/')
# from Evaluate import load_and_eval
# load_and_eval('saved_models/xgboost_model_THCA.pkl', 'test_data_indices_xgboost_THCA', 'THCA_load_test', 'THCA', remove_inf = False)

def load_and_eval_keras(model_path, indices_path, save_name, cancer_code, remove_inf):
    from keras.models import load_model
    model = load_model(model_path)
    
    Xtest, ytest = load_data_from_indices(indices_path, cancer_code, remove_inf)
    import numpy as np
    Xtest = np.expand_dims(Xtest, axis=2)
    
    # evaluate
    y_pred = model.predict(Xtest, batch_size=Xtest.shape[0])
  

    import Evaluate

    label = save_name+cancer_code

    from Evaluate import print_evaluation, plot_curve
    print_evaluation(None, None, ytest, label, predictions=np.argmax(y_pred, axis=1), prob_predictions=y_pred) # fitted and m_values are none because we are providing predictions

    num_classes = len(np.unique(ytest, axis=0))
    
    predictions = np.argmax(y_pred, axis=1)
#     make_labelled_conf_mat(ytest, predictions, num_classes, save_name)
    
    plot_curve('roc', ytest, y_pred, num_classes, label, cancer_type = cancer_code)
    plot_curve('precision_recall', ytest, y_pred, num_classes, label, cancer_type = cancer_code)

    
# calls evaluation functions for external data, using a multiclass model
def evaluate_external_multiclass(m_values, diagnoses, model, cancer_type, save_name, save_folder = '', no_roc = True):
    
    cancer_types = ['Normal', 'BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA', 'UCEC']
    cancer_type_number = cancer_types.index(cancer_type)
        
    # convert diagnoses to multiclass:
    import numpy as np
    diagnoses_multi = np.where(diagnoses == 1, cancer_type_number, 0) # only works when diagnoses contains one cancer type (and normal)

    from Evaluate import print_evaluation
    
    if 'ffNN' in save_name:
        y_pred = model.predict(m_values, batch_size=m_values.shape[0])
        print_evaluation(None, None, diagnoses_multi, save_name, predictions=np.argmax(y_pred, axis=1), prob_predictions=y_pred, no_roc=True, cm_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13], small_text = True, save_folder = save_folder) # fitted and m_values are none because we are providing predictions

    else:
        print_evaluation(model, m_values, diagnoses_multi, save_name, no_roc=no_roc, cm_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13], small_text = True, save_folder = save_folder)

    confidence = model.predict_proba(m_values)
    import numpy as np
    num_classes = 14
    print("num classes is: ", num_classes)

    from Evaluate import plot_curve
    plot_curve('roc', diagnoses_multi, confidence, num_classes, save_name, cancer_type = cancer_type, save_folder = save_folder)
    plot_curve('precision_recall', diagnoses_multi, confidence, num_classes, save_name, cancer_type = cancer_type, save_folder = save_folder)


# given a threshold and confidence values, works out metrics for a specific class (c) like the precision, recall, etc
# useful for checking and understanding pr-curves and ROC curves
def get_metrics_given_threshold(threshold, confidence, c, big_diagnoses):
    import numpy as np
    predicted_vals = np.where(confidence[:,c] > threshold)[0]
    print("predicted positive values: ")
    print(predicted_vals)

    true_vals = np.where(np.array(big_diagnoses) == c)[0]
    print("true positive values: ")
    print(true_vals)

    tp = len(set(predicted_vals).intersection(set(true_vals)))
    fp = len(set(predicted_vals).difference(set(true_vals)))
    fn = len(set(true_vals).difference(set(predicted_vals)))
    print("tp: ", tp)
    print("fp: ", fp)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    tp_rate = tp/len(true_vals)
    fp_rate = fp/np.sum((np.array(big_diagnoses) != c))

    print("precision", precision)
    print("recall", recall)
    print("tp_rate ", tp_rate)
    print("fp_rate", fp_rate)