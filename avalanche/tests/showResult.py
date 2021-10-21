import pandas as pd
import matplotlib.pyplot as plt




def plotResults(results, n_exp):
    loss_str = 'Loss_Stream/eval_phase/test_stream/'
    accuracy_str = 'Top1_Acc_Stream/eval_phase/test_stream/'
    df_acc = pd.DataFrame()
    df_loss = pd.DataFrame()
    for exp in range(n_exp):
        #print('training exp:', exp)
        acc = []
        loss = []
        for i in range(n_exp):

            s = 'Task00' +str(i)
            acc.append(results[exp][accuracy_str+s])
            loss.append(results[exp][loss_str+s])
          
        df_acc['after exp_' + str(exp)] = acc
        df_loss['after exp_' + str(exp)] = loss
        
    
    res_acc = list(df_acc.mean())
    res_loss = list(df_loss.mean())
    print('\n####################################################################################################################################')
    print('\nThe following table contains acuracy and loss values obtained on test set after training x Experience. Each column stays for x esp trained.')
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