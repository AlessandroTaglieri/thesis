from typing import List
from avalanche.training.strategies import EWC, Replay, Naive
from avalanche.models import SimpleMLP, SimpleCNN
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from models.CNN import CNN
from benchmarks.classic.splitESC50 import CLEsc50, CLEsc50_v2
import pandas as pd
import wandb
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, forward_transfer_metrics, bwt_metrics
    



def  test_EWC_CNN_nc():

    splitEsc =  CLEsc50(n_experiences=10, seed=123, return_task_id=True,fixed_class_order=None,shuffle=True)

    model = CNN()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    interactive_logger = InteractiveLogger()

    config = wandb.config
    wandb_logger = WandBLogger(project_name="tesi", run_name="ewc_nc", 
                               config=config)
    eval_plugin = EvaluationPlugin(
            accuracy_metrics(
                minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            bwt_metrics(experience=True, stream=True),
            forward_transfer_metrics(stream=True),
            loggers=[interactive_logger, wandb_logger])
    cl_strategy = EWC(
        model, optimizer, criterion, ewc_lambda=0.4,
        train_mb_size=15, train_epochs=25, eval_mb_size=15,evaluator=eval_plugin, eval_every = 0)

    

    train_stream = splitEsc.train_stream
    test_stream = splitEsc.test_stream

    print('Starting experiment...')
    results = []
    for experience in train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience,
                                eval_streams=[test_stream])
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(test_stream))
        #print(cl_strategy.eval(scenario.test_stream))
        print('**************************************')

    return results

def  test_EWC_CNN():

    splitEsc =  CLEsc50(n_experiences=10, seed=123, return_task_id=True,fixed_class_order=None,shuffle=True)

    model = CNN()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    cl_strategy = EWC(
        model, optimizer, criterion, ewc_lambda=0.4,
        train_mb_size=15, train_epochs=10, eval_mb_size=15
    )

    train_stream = splitEsc.train_stream
    test_stream = splitEsc.test_stream

    print('Starting experiment...')
    results = []
    for experience in train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(test_stream))
        #print(cl_strategy.eval(scenario.test_stream))
        print('**************************************')

    return results

def  test_EWC_CNN_v2():

    splitEsc =  CLEsc50(n_experiences=10, seed=123, return_task_id=True,fixed_class_order=None,shuffle=True)

    model = CNN()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    cl_strategy = EWC(
        model, optimizer, criterion, ewc_lambda=0.4,
        train_mb_size=15, train_epochs=25, eval_mb_size=15
    )

    train_stream = splitEsc.train_stream
    test_stream = splitEsc.test_stream

    print('Starting experiment...')
    results = []
    for experience in train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(test_stream))
        #print(cl_strategy.eval(scenario.test_stream))
        print('**************************************')

    return results


def  test_EWC_CNN_v3():

    splitEsc =  CLEsc50(n_experiences=10, seed=123, return_task_id=True,fixed_class_order=None,shuffle=True)

    model = CNN()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    cl_strategy = EWC(
        model, optimizer, criterion, ewc_lambda=0.4,
        train_mb_size=15, train_epochs=10, eval_mb_size=15
    )

    train_stream = splitEsc.train_stream
    test_stream = splitEsc.test_stream

    print('Starting experiment...')
    results = []
    for experience in train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(test_stream))
        #print(cl_strategy.eval(scenario.test_stream))
        print('**************************************')

    return results

def  test_EWC_CNN_v4():

    splitEsc =  CLEsc50(n_experiences=10, seed=123, return_task_id=True,fixed_class_order=None,shuffle=True)

    model = CNN()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    cl_strategy = EWC(
        model, optimizer, criterion, ewc_lambda=0.4,
        train_mb_size=32, train_epochs=10, eval_mb_size=32
    )

    train_stream = splitEsc.train_stream
    test_stream = splitEsc.test_stream

    print('Starting experiment...')
    results = []
    for experience in train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(test_stream))
        #print(cl_strategy.eval(scenario.test_stream))
        print('**************************************')

    return results



      
       