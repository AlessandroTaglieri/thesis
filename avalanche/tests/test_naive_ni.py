from typing import List
from avalanche.training.strategies import EWC, Replay, Naive
from avalanche.models import SimpleMLP, SimpleCNN
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from models.CNN import CNN
from benchmarks.classic.splitESC50_v2 import CLEsc50, CLEsc50_v2
import pandas as pd
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, forward_transfer_metrics, bwt_metrics
    


def test_Naive_CNN_ni():

    splitEsc =  CLEsc50(n_experiences=10, seed=123, return_task_id=True,balance_experiences=True,shuffle=True)

    model = CNN(num_classes=50)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
            accuracy_metrics(
                minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            bwt_metrics(experience=True,stream=True),
            #forward_transfer_metrics(experience=True,stream=True),
            loggers=[interactive_logger])
    cl_strategy = Naive(model, optimizer, criterion, train_mb_size=15, train_epochs=25, eval_mb_size=15,evaluator=eval_plugin)#, eval_every = 0)

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