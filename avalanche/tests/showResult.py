from numpy.core.fromnumeric import mean
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
    df_acc2 = pd.DataFrame()
    df_loss2 = pd.DataFrame()
    df_loss = pd.DataFrame()
    df_btw = pd.DataFrame()
    acc2_avg = []
    loss2_avg = []
    bwt_avg = []
    fwt_avg = []
    forgetting_avg = []
    for exp in range(n_exp):
        
        acc = []
        loss = []
        btw = []
        acc2 = []
        loss2 = []
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

                ########
                
                if i <= exp:
                    
                    s3 = 'Task00' +str(i)
                    acc2.append(results[exp][accuracy_str+s3])
                    loss2.append(results[exp][loss_str+s3])
                else:
                    loss2.append(0)
                    acc2.append(0)
            else:
                s = 'Task000'
                
            
            acc.append(results[exp][accuracy_str+s])
            loss.append(results[exp][loss_str+s])
        if nc:
          df_btw['after exp_' + str(exp)] = btw
        df_acc['after exp_' + str(exp)] = acc
        df_acc2['after exp_' + str(exp)] = acc2
        df_loss2['after exp_' + str(exp)] = loss2
        df_loss['after exp_' + str(exp)] = loss

        acc2_avg.append(mean(acc2[0:exp+1]))
        loss2_avg.append(mean(loss2[0:exp+1]))
        
    
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
    if nc:
        fig.savefig('plotMetrics/singlePlot/task_incremental/bwt/'+name+'_BWTmetric.png', dpi=fig.dpi)
    else:
        fig.savefig('plotMetrics/singlePlot/domain_incremental/bwt/'+name+'_BWTmetric.png', dpi=fig.dpi)
    
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
    if nc:
        fig.savefig('plotMetrics/singlePlot/task_incremental/forgetting/'+name+'_ForgettingMetric.png', dpi=fig.dpi)
    else:
        
        fig.savefig('plotMetrics/singlePlot/domain_incremental/forgetting/'+name+'_ForgettingMetric.png', dpi=fig.dpi)
    

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
    if nc:
        fig.savefig('plotMetrics/singlePlot/task_incremental/fwt/'+name+'_FWTmetric.png', dpi=fig.dpi)

    else:
        fig.savefig('plotMetrics/singlePlot/domain_incremental/fwt/'+name+'_FWTmetric.png', dpi=fig.dpi)

    print('\n####################################################################################################################################')
    print('\nThe following table contains accuracy and loss values obtained on test set after training x Experience. Each column stays for x exp trained.')
    columns = ['Exp_' +str(i) for i in range(n_exp)]
    
    if nc:
        list_values = [acc2_avg, loss2_avg, res_acc, res_loss]
        list_labels = ['Accuracy avg on past task','Loss avg on past task',' Global Accuracy', 'Loss']
    else:
        list_values = [res_acc, res_loss]
        list_labels = [' Global Accuracy', 'Loss']
    df = pd.DataFrame(list_values, columns = columns)
    df.index = list_labels

    display(df)
    
    
    if nc:

    
        print('\n\n####################################################################################################################################')
        print('\nThe following plot shows the behaviour of the accuracy in test set during training over all experience trained beforethe current experience')
        
        
        
        
        fig = plt.figure(figsize=(12, 8), dpi=80)
        plt.plot(x_exp, acc2_avg, label = 'accuracy')
        plt.title('Accuracy avg for first x Experiences')
        plt.xlabel('Experience')
        plt.ylabel('Accuracy metric ')
        plt.xticks(x_exp)
        plt.ylim(0.2,0.7)
        plt.grid()
        plt.show()
        
        fig.savefig('plotMetrics/singlePlot/task_incremental/accuracy/'+name+'_AccuracyMetric_before.png', dpi=fig.dpi)
        


        print('\n\n####################################################################################################################################')
        print('\nThe following plot shows the behaviour of the loss in test set during training over all experience trained beforethe current experience')
        
        
        
        
        fig = plt.figure(figsize=(12, 8), dpi=80)
        plt.plot(x_exp, loss2_avg, label = 'accuracy')
        plt.title('Loss avg for first x Experiences')
        plt.xlabel('Experience')
        plt.ylabel('Loss metric ')
        plt.xticks(x_exp)
        
        plt.grid()
        plt.show()
        fig.savefig('plotMetrics/singlePlot/task_incremental/loss/'+name+'_LossMetric_before.png', dpi=fig.dpi)
       



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
    if nc:
        fig.savefig('plotMetrics/singlePlot/task_incremental/accuracy/'+name+'_AccuracyMetric.png', dpi=fig.dpi)
    else:
        fig.savefig('plotMetrics/singlePlot/domain_incremental/accuracy/'+name+'_AccuracyMetric.png', dpi=fig.dpi)
    print('\n\n\n####################################################################################################################################')
    print('\nThe following plot shows the behaviour of the loss in test set during training over all experience')
    
    fig = plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(x_exp,res_loss, label = 'loss')
    plt.title('Loss after x exp')
    plt.xlabel('Experience')
    plt.ylabel('Loss metric')
    plt.grid()
    plt.show()
    if nc:

        fig.savefig('plotMetrics/singlePlot/task_incremental/loss/'+name+'_LossMetric.png', dpi=fig.dpi)
    else:
        fig.savefig('plotMetrics/singlePlot/domain_incremental/loss/'+name+'_LossMetric.png', dpi=fig.dpi)  
    result = {}
    result['accuracy'] = res_acc
    result['loss'] = res_loss
    result['bwt'] = bwt_avg
    result['forgetting'] = forgetting_avg
    result['fwt'] = fwt_avg
    result['accuracy_before'] = acc2_avg
    result['loss_before'] = loss2_avg


    return result


def compareResults(x, y, title, ylabel, list_labels,name,nc):
    fig = plt.figure(figsize=(12, 8), dpi=80)
    for index in range(len(y)):
        plt.plot(x, y[index], label = list_labels[index])
    
    plt.title(title)
    plt.xlabel('Experience')
    plt.ylabel(ylabel)
    
    plt.xticks(x)
    plt.legend()
    plt.grid()
    plt.show()
    
    fig2, ax = plt.subplots(figsize=(12, 8), dpi=80)
   
    data = pd.DataFrame( {list_labels[0]: y[0],
     list_labels[1]: y[1],
     list_labels[2]: y[2]
    })
    plt.violinplot(data,showmeans=True,vert=True)
    plt.title(title)
    plt.xlabel('Continual learning technique')
    plt.ylabel(ylabel)
    
    ax.set_xticklabels(['',list_labels[0],'', list_labels[1],'',list_labels[2]])

    
    plt.grid()
    plt.show()
    if nc:

        fig2.savefig('plotMetrics/overview/task_incremental/overview2_'+name+'.png', dpi=fig2.dpi)
        fig.savefig('plotMetrics/overview/task_incremental/overview_'+name+'.png', dpi=fig.dpi)
    else:
        fig2.savefig('plotMetrics/overview/domain_incremental/overview2_'+name+'.png', dpi=fig2.dpi)
        
        fig.savefig('plotMetrics/overview/domain_incremental/overview_'+name+'.png', dpi=fig.dpi)