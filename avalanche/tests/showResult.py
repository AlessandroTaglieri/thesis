import pandas as pd
import matplotlib.pyplot as plt




def plotResults2(results, n_exp, nc = True):
    loss_str = 'Loss_Stream/eval_phase/test_stream/'
    accuracy_str = 'Top1_Acc_Stream/eval_phase/test_stream/'
    btw_str = 'ExperienceBWT/eval_phase/test_stream/'
    btw_avg_str = 'StreamBWT/eval_phase/test_stream'
    df_acc = pd.DataFrame()
    df_loss = pd.DataFrame()
    df_btw = pd.DataFrame()
    btw_avg = []
    for exp in range(n_exp):
        #print('training exp:', exp)
        acc = []
        loss = []
        btw = []
        btw_avg.append(results[exp][btw_avg_str])
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

        df_btw['after exp_' + str(exp)] = btw
        df_acc['after exp_' + str(exp)] = acc
        df_loss['after exp_' + str(exp)] = loss
    print('\n####################################################################################################################################')
    print('\nThe following table contains BTW on test set after training x Experience. Each column stays for x exp trained.')
    
    display(df_btw)

    print('\n\n####################################################################################################################################')
    print('\nThe following plot shows the behaviour of the BWT in test set during training over all experience')
    
    
    x_exp = [i for i in range(n_exp)]
    
    fig = plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x_exp, btw_avg, label = 'BTW')
    plt.title('BTW after x exp')
    plt.xlabel('Experience')
    plt.ylabel('BTW')
    plt.xticks(x_exp)
    plt.grid()
    plt.show()
    res_acc = list(df_acc.mean())
    res_loss = list(df_loss.mean())
    print('\n####################################################################################################################################')
    print('\nThe following table contains accuracy and loss values obtained on test set after training x Experience. Each column stays for x exp trained.')
    columns = ['Exp_' +str(i) for i in range(n_exp)]
    df = pd.DataFrame([res_acc, res_loss], columns = columns)
    df.index = ['Acuracy', 'Loss']
    display(df)
    
    print('\n\n####################################################################################################################################')
    print('\nThe following plot shows the behaviour of the accuracy in test set during training over all experience')
    
    x_exp = [i for i in range(n_exp)]
    
    fig = plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x_exp, res_acc, label = 'accuracy')
    plt.title('Accuracy after x exp')
    plt.xlabel('Experience')
    plt.ylabel('Accuracy')
    plt.xticks(x_exp)
    plt.ylim(0,0.7)
    plt.grid()
    plt.show()
    
    print('\n\n\n####################################################################################################################################')
    print('\nThe following plot shows the behaviour of the loss in test set during training over all experience')
    
    fig = plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x_exp,res_loss, label = 'loss')
    plt.title('Loss after x exp')
    plt.xlabel('Experience')
    plt.ylabel('Loss')
    plt.show()

    return [x_exp, res_acc]