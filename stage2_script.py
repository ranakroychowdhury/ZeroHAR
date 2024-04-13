import utils, argparse, warnings, sys, shutil, torch, os, numpy as np, math
from itertools import chain
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--expt_name', type=str, default='code_v4_two_stages')
parser.add_argument('--dataset', type=str, default='uschad')
parser.add_argument('--fold', type=str, default='fold0')
parser.add_argument('--zsl', action='store_true')
parser.add_argument('--device', type=str, default='5')
parser.add_argument('--pretrained', action='store_true')

parser.add_argument('--pooling', type=str, default='last time', help="['anything', 'mean', 'att', 'bert']")
parser.add_argument('--base_path', type=str, default='/home/ranak/gzsl_project/')
args = parser.parse_args()


def main():
    prop = utils.get_parameters(args.dataset, args.fold, args.zsl, args.base_path + 'metadata/')
    prop['expt_name'] = args.expt_name
    prop['base_path'] = args.base_path
    prop['dataset'], prop['fold'], prop['pooling'] = args.dataset, args.fold, args.pooling
    prop['zsl'] = True if args.zsl else False
    prop['stage'] = '2'
    prop['device'] = torch.device('cuda:' + args.device if torch.cuda.is_available() else "cpu")
    prop['pretrained'] = args.pretrained

    print('dataset: ' + prop['dataset'] + ', ' + prop['fold'] + ', pooling: ' + prop['pooling'] + ', ZSL: ' + str(prop['zsl']) + ', GPU: ' + str(prop['device']))    
    for classid in prop['classid_to_word_embedding']:
        print(str(classid) + ' ' + str(prop['classid_to_word_embedding'][classid].shape))

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
    print('Total classes: ' + str(prop['nclasses']) + ', Train classes: ' + str(prop['num_train_classes']) + ', Test classes: ' + str(prop['num_test_classes']))
    print('Data loading complete...')
    
    print('Initializing model...')
    model, optimizer_time_series, optimizer_word_embedding, criterion, best_model, best_optimizer_time_series, best_optimizer_word_embedding = utils.initialize_training(prop)
    print('Model intialized...')

    if prop['pretrained']:
        
        print('Loading pretrained model and optimizer...')
        model_file = prop['base_path'] + prop['expt_name'] + '/models/' + prop['dataset'] + '_' + str(prop['zsl']) + '_' + str(prop['fold']) + '_stage1_model_optimizer.pth'
        
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(prop['device'])

        optimizer_time_series.load_state_dict(checkpoint['optimizer_time_series_state_dict'])
        optimizer_word_embedding.load_state_dict(checkpoint['optimizer_word_embedding_state_dict'])

        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model.to(prop['device'])

        best_optimizer_time_series.load_state_dict(checkpoint['optimizer_time_series_state_dict'])
        best_optimizer_word_embedding.load_state_dict(checkpoint['optimizer_word_embedding_state_dict'])
        print('Pretrained model and optimizer loaded...')

    print('Training start...')
    utils.stage2_train_and_test(model, optimizer_time_series, optimizer_word_embedding, criterion, best_model, best_optimizer_time_series, best_optimizer_word_embedding, X_train, y_train, X_test, y_test, prop)



if __name__ == "__main__":
    main()