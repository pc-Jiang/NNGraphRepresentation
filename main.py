import argparse
import experiments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--analyze', nargs='+', help='Analyze experiments', default=[])
    parser.add_argument('-c', '--cluster', action='store_true', help='Use batch submission on cluster')
    args = parser.parse_args()
    experiments2analyze = args.analyze
    use_cluster = args.cluster

    for exp in experiments2analyze:
        if exp in dir(experiments):
            getattr(experiments, exp)()
