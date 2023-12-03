import argparse
import experiments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--analyze', nargs='+', help='Analyze experiments', default=[])
    args = parser.parse_args()
    experiments2analyze = args.analyze

    for exp in experiments2analyze:
        if exp in dir(experiments):
            getattr(experiments, exp)()
