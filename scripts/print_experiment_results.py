import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    path_to_results = f'outputs/{args.experiment_name}/eval.log'
    
    experiment_settings = args.experiment_name.split('_')
    print(f'Results for experiment {args.experiment_name}')

    with open(path_to_results, 'r') as results_f:
        results_text = results_f.readlines()

    for log_line in results_text:
        log_line = log_line.strip()
        if 'for metric' in log_line:
            print(log_line.split(' - ')[1])
        if 'score: ' in log_line:
            #if "+" in log_line:
            #    print(log_line.split(': ')[1])
            #else:
            print(log_line.split(': ')[1])
        if 'upper_bound' in log_line:
            print('---')


