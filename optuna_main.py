import optuna
from main import train
from opt import args
import yaml

cut_l = {'MUTAG': 10, 'ENZYMES': 25, 'PROTEINS': 50, 'REDDIT-MULTI-5K': 15, 'NCI1': 25, 'BZR': 13, 'DD': 65,
        'REDDIT-BINARY': 13, 'PTC_MR': 20}
cut_h = {'MUTAG': 15, 'ENZYMES': 35, 'PROTEINS': 64, 'REDDIT-MULTI-5K': 24, 'NCI1': 42, 'BZR': 19, 'DD': 82,
        'REDDIT-BINARY': 19, 'PTC_MR': 30}


def optuna_train(trial: optuna.trial.Trial):
    with open('config.yml', 'r') as file:
        data = yaml.safe_load(file)
    config = data['dataset'][args.dataset]
    acc = train(
        lr=trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        hid_paths=trial.suggest_int('hid_paths', 200, 800, step=100),
        norm=config['norm'],
        cutoff=trial.suggest_int('cutoff', cut_l[args.dataset], cut_h[args.dataset], step=1),
        # cutoff= None , #IM-B,IM-M,COLLAB
        dropout=trial.suggest_float('dropout', 0.0, 0.4, step=0.05),
        norm_attr=config['norm_attr'])
    return acc


if __name__ == '__main__':
    study: optuna.study.Study = optuna.create_study(study_name='{}-ICML'.format(args.dataset),
                                                    storage="sqlite:///optuna/final.sqlite3",
                                                    direction="maximize",
                                                    load_if_exists=True,
                                                    sampler=optuna.samplers.RandomSampler())
    study.optimize(optuna_train, n_trials=10000)
