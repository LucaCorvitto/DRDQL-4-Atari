import argparse
from train import train
from train_drdqn import train as train_drdqn 
from train_drqn import train as train_drqn 
from evaluate import evaluate
from evaluate_rnn import evaluate as evaluate_rnn

def main():
    
    parser = argparse.ArgumentParser()
    
    # set hyperparameter
    
    args.add_argument('-t','--training', action='store_true', default=False)
    args.add_argument('-e', '--evaluating', action='store_true', default=False)
    args.add_argument('-r', '--rendering', action='store_true', default=False)
    args.add_argument('-tot', '--total_steps', type=int, default=200000)
    args.add_argument('-n', '--network', type=str, default='DRDQN')
    args.add_argument('-env', '--environment', type=str, default='SpaceInvaders-v5')
    
    args = parser.parse_args()
    training = args.training
    evaluating = args.evaluating
    rendering = args.rendering
    network_name = args.network
    env_name = args.env
    total_steps = args.total_steps

    trained = False

    if training:
        if network_name == 'DRDQN':
            train_drdqn(env_name, network_name, total_steps)
        elif network_name == 'DRQN':
            train_drqn(env_name, network_name, total_steps)
        else:
            train(env_name, network_name, total_steps)
        
        trained = True

    if evaluating:
        if 'R' in network_name:
            evaluate_rnn(env_name, network_name, trained, rendering, total_steps)
        else:
            evaluate(env_name, network_name, trained, rendering, total_steps)

    if not training and not evaluating:
        print('Please specify if the program has to train by adding in the command "-t" and/or evaluate with "-e"')


if __name__ == "__main__":
    main()