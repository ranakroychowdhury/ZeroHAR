import utils, argparse, warnings, sys, shutil, torch, os, numpy as np, math
from itertools import chain
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--expt_name', type=str, default='code_v4_two_stages')
parser.add_argument('--dataset', type=str, default='uschad')
parser.add_argument('--fold', type=str, default='fold1')
parser.add_argument('--zsl', action='store_true')
parser.add_argument('--device', type=str, default='7')

parser.add_argument('--pooling', type=str, default='last time', help="['anything', 'mean', 'att', 'bert']")
parser.add_argument('--base_path', type=str, default='/home/ranak/gzsl_project/')
args = parser.parse_args()


def main():
    prop = utils.get_parameters(args.dataset, args.fold, args.zsl, args.base_path + 'metadata/')
    prop['expt_name'] = args.expt_name
    prop['base_path'] = args.base_path
    prop['dataset'], prop['fold'], prop['pooling'] = args.dataset, args.fold, args.pooling
    prop['zsl'] = True if args.zsl else False
    prop['stage'] = '1'
    prop['device'] = torch.device('cuda:' + args.device if torch.cuda.is_available() else "cpu")

    print('dataset: ' + prop['dataset'] + ', ' + prop['fold'] + ', pooling: ' + prop['pooling'] + ', ZSL: ' + str(prop['zsl']) + ', GPU: ' + str(prop['device']))    

    print('Data loading start...')
    data_path = prop['base_path'] + 'data/' + prop['dataset'] + '/'
    if prop['zsl']:
        X_train, y_train, X_test, y_test = utils.data_loader_zsl(data_path, prop)
    else:
        X_train, y_train, X_test, y_test = utils.data_loader_gzsl(data_path, prop)

    mean, std = utils.mean_standardize_fit(X_train)
    X_train, X_test = utils.mean_standardize_transform(X_train, mean, std), utils.mean_standardize_transform(X_test, mean, std)

    prop['seq_len'], prop['input_size'] = X_train.shape[1], X_train.shape[2]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print('Data loading complete...')
    
    print('Initializing model...')
    model, optimizer_time_series, optimizer_word_embedding, criterion, best_model, best_optimizer_time_series, best_optimizer_word_embedding = utils.initialize_training(prop)
    print('Model intialized...')

    print('Training start...')
    utils.stage1_training(model, optimizer_time_series, optimizer_word_embedding, criterion, best_model, best_optimizer_time_series, best_optimizer_word_embedding, X_train, prop)
    print('Training complete...')


if __name__ == "__main__":
    main()



'''
"Opp_g": {"fold0": {}, 
            "fold1": {"0": "open door", "3": "close fridge", "4": "open dishwasher", "7": "close drawer"}, 
            "fold2": {"1": "close door", "2": "open fridge", "5": "close dishwasher", "6": "open drawer"}, 
            "fold3": {"8": "clean table", "9": "drink from cup", "10": "toggle switch"}
'''