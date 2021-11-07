import pandas as pd
import matplotlib.pyplot as plt




def plotResults(results, n_exp, name, nc = True):
    loss_str = 'Loss_Stream/eval_phase/test_stream/'
    accuracy_str = 'Top1_Acc_Stream/eval_phase/test_stream/'
    btw_str = 'ExperienceBWT/eval_phase/test_stream/'
    bwt_avg_str = 'StreamBWT/eval_phase/test_stream'
    fwt_avg_str = 'StreamForwardTransfer/eval_phase/test_stream'
    forgetting_avg_str = 'StreamForgetting/eval_phase/test_stream'
    df_acc = pd.DataFrame()
    df_loss = pd.DataFrame()
    df_btw = pd.DataFrame()
    bwt_avg = []
    fwt_avg = []
    forgetting_avg = []
    for exp in range(n_exp):
        
        acc = []
        loss = []
        btw = []
        fwt_avg.append(results[exp][fwt_avg_str])
        forgetting_avg.append(results[exp][forgetting_avg_str])
        bwt_avg.append(results[exp][bwt_avg_str])
        for i in range(n_exp):

            if nc:
                
                s = 'Task00' +str(i)
                if exp == 0:
                    btw.append(0)
                elif i < exp:
                    
                    s2 = 'Task00' +str(i) + '/Exp00' + str(i)
                    
                    btw.append(results[exp][btw_str+s2])
                else:
                    btw.append(0)
            else:
                s = 'Task000'
                
            
            acc.append(results[exp][accuracy_str+s])
            loss.append(results[exp][loss_str+s])
        if nc:
          df_btw['after exp_' + str(exp)] = btw
        df_acc['after exp_' + str(exp)] = acc
        df_loss['after exp_' + str(exp)] = loss
    if nc:
        print('\n####################################################################################################################################')
        print('\nThe following table contains BTW on test set after training x Experience. Each column stays for x exp trained.')
        
        display(df_btw)

    print('\n\n####################################################################################################################################')
    print('\nThe following plot shows the behaviour of the average BWT in test set during training over all experience')
    
    
    x_exp = [i for i in range(n_exp)]
    
    fig = plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(x_exp, bwt_avg, label = 'BWT')
    plt.title('BWT after x exp')
    plt.xlabel('Experience')
    plt.ylabel('BWT metric')
    plt.xticks(x_exp)
    plt.grid()
    plt.show()
    res_acc = list(df_acc.mean())
    res_loss = list(df_loss.mean())
    fig.savefig('plotMetrics/'+name+'_BWTmetric.png', dpi=fig.dpi)
    print('\n\n####################################################################################################################################')
    print('\nThe following plot shows the behaviour of the average FORGETTING in test set during training over all experience')
    
    
   
    
    fig = plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(x_exp, forgetting_avg, label = 'Forgetting')
    plt.title('Forgetting after x exp')
    plt.xlabel('Experience')
    plt.ylabel('Forgetting metric')
    plt.xticks(x_exp)
    plt.grid()
    plt.show()
    fig.savefig('plotMetrics/'+name+'_ForgettingMetric.png', dpi=fig.dpi)
    

    print('\n\n####################################################################################################################################')
    print('\nThe following plot shows the behaviour of the average FWT in test set during training over all experience')
    
    
  
    
    fig = plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(x_exp, fwt_avg, label = 'FWT')
    plt.title('FWT after x exp')
    plt.xlabel('Experience')
    plt.ylabel('FWT metric')
    plt.xticks(x_exp)
    plt.grid()
    plt.show()
    fig.savefig('plotMetrics/'+name+'_FWTmetric.png', dpi=fig.dpi)

    print('\n####################################################################################################################################')
    print('\nThe following table contains accuracy and loss values obtained on test set after training x Experience. Each column stays for x exp trained.')
    columns = ['Exp_' +str(i) for i in range(n_exp)]
    df = pd.DataFrame([res_acc, res_loss], columns = columns)
    df.index = ['Acuracy', 'Loss']
    display(df)
    
    print('\n\n####################################################################################################################################')
    print('\nThe following plot shows the behaviour of the accuracy in test set during training over all experience')
    
    
    res_acc = list(df_acc.mean())
    res_loss = list(df_loss.mean())
    fig = plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(x_exp, res_acc, label = 'accuracy')
    plt.title('Accuracy after x exp')
    plt.xlabel('Experience')
    plt.ylabel('Accuracy metric')
    plt.xticks(x_exp)
    plt.ylim(0,0.7)
    plt.grid()
    plt.show()
    fig.savefig('plotMetrics/'+name+'_AccuracyMetric.png', dpi=fig.dpi)
    print('\n\n\n####################################################################################################################################')
    print('\nThe following plot shows the behaviour of the loss in test set during training over all experience')
    
    fig = plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(x_exp,res_loss, label = 'loss')
    plt.title('Loss after x exp')
    plt.xlabel('Experience')
    plt.ylabel('Loss metric')
    plt.show()
    fig.savefig('plotMetrics/'+name+'_LossMetric.png', dpi=fig.dpi)

    result = {}
    result['accuracy'] = res_acc
    result['loss'] = res_loss
    result['bwt'] = bwt_avg
    result['forgetting'] = forgetting_avg
    result['fwt'] = fwt_avg


    return result


def compareResults(x, y, title, ylabel, list_labels,name):
    fig = plt.figure(figsize=(12, 8), dpi=80)
    for index in range(len(y)):
        plt.plot(x, y[index], label = list_labels[index])
    
    plt.title('Accuracy after x exp')
    plt.xlabel('Experience')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x)
    
    plt.grid()
    plt.show()
    fig.savefig('plotMetrics/compare_'+name+'.png', dpi=fig.dpi)