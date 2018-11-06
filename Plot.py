
import matplotlib.pyplot as plt
import numpy as np

ds_name = [["AD","NC"],["AD","EMCI"],["AD","LMCI"],["LMCI","NC"],["LMCI","EMCI"],["EMCI","NC"]]
plot_history_path="H:\Google Drive\plot"
plot_data_type= ["ROC_AUC", "evaluation", "training"]

def get_plot_data(history_path,prefix):
    whole_history_list = list()
    whole_data_list = list()
    for i in range(len(ds_name)):
        history_list = list()
        data_list = list()
        history_txt_prefix="{}\\{}_history_{}_vs_{}".format(str(history_path),str(prefix),str(ds_name[i][0]),str(ds_name[i][1]))
        for j in range(1,len(ds_name)):
            history_txt="{}_fold_index_{}.txt".format(str(history_txt_prefix),str(j))
            history_list.append(history_txt)
            #print(history_txt)
            data = np.loadtxt(history_txt, delimiter=',', usecols = 1)
            data_list.append(data)
        whole_history_list.append(history_list)
        whole_data_list.append(data_list)
    return np.asanyarray(whole_data_list)

def plot_ROC_AUC(ROC_AUC,prefix):
    mean = list()
    std = list()
    #for i in range(ROC_AUC.shape[-1]):
    for i in range(70):
        mean.append(np.mean(ROC_AUC[ds_name.index(prefix), :, i]))
        std.append(np.std(ROC_AUC[ds_name.index(prefix), :, i]))
    mean = np.asanyarray(mean)
    std = np.asanyarray(std)

    plt.plot(mean)
    plt.fill_between(list(range(len(mean))), mean + std, mean - std, color='gray', alpha=0.2)
    #plt.title('AD vs NC classification ROC AUC')
    plt.title('{} vs {} classification ROC AUC'.format(str(prefix[0]), str(prefix[1])))
    plt.ylabel('ROC AUC')
    plt.xlabel('epoch')
    plt.legend(['Validation ROC AUC mean', 'Validation ROC AUC std'], loc='lower left')
    plt.show()
       
ROC_AUC = get_plot_data(plot_history_path,plot_data_type[0])
plot_ROC_AUC(ROC_AUC,ds_name[0])