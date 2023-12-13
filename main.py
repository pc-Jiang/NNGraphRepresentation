import argparse
import experiments
from experiments import run_experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--analyze', nargs='+', help='Analyze experiments', default=[])
    args = parser.parse_args()
    experiments2analyze = args.analyze

    for exp in experiments2analyze:
        if exp in dir(experiments):
            if exp != 'replot':
                exp_configs = getattr(experiments, exp)()
                run_experiment(exp_configs)
            else:
                getattr(experiments, exp)()
