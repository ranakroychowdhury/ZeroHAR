import warnings, json, pickle, torch, math, os, random, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import architecture
# import editdistance
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")



# loading hyperparameters
def get_parameters(dataset, fold, zsl, base_path):

    hyperparameters_path = base_path + 'hyperparameters.json'
    with open(hyperparameters_path, 'r') as openfile:
        # Reading from json file
        hyperparameters = json.load(openfile)
    hyperparameters = hyperparameters[dataset]

    folds_classid_to_labelname_path = base_path + 'folds_classid_to_labelname_mapping.json'
    with open(folds_classid_to_labelname_path, 'r') as openfile:
        # Reading from json file
        folds_classid_to_labelname_mapping = json.load(openfile)


    #---------------------- CLASS ID -> LABEL NAME ----------------------------#
    classid_to_labelname_path = base_path + 'classid_to_labelname_mapping.json'
    with open(classid_to_labelname_path, 'r') as openfile:
        # Reading from json file
        classid_to_labelname_mapping = json.load(openfile)
    #---------------------- CLASS ID -> LABEL NAME ----------------------------#


    labelname_to_classid_path = base_path + 'labelname_to_classid_mapping.json'
    with open(labelname_to_classid_path, 'r') as openfile:
        # Reading from json file
        labelname_to_classid_mapping = json.load(openfile)


    #---------------------LABEL NAME -> LABEL EMBEDDING ----------------------------#
    all_labeldescriptions_to_word_embedding_path = base_path + 'gpt4_label_description_variations_imagebind_embedding.pickle'
    with open(all_labeldescriptions_to_word_embedding_path, 'rb') as f:
        # labelname_to_word_embedding is a dictionary with dataset name as the keys
        # Each dataset key is also a dictionary with labelnames as the keys and their corresponding word embedding as values
        all_labeldescriptions_to_word_embedding = pickle.load(f)[dataset]

    labelname_to_word_embedding = {} # -> dictionary with label name as keys and a torch of size [10, 1024] as the value, 10 different descriptions, 1024 Imagebind embedding size
    for label in all_labeldescriptions_to_word_embedding:
        
        labelname_to_word_embedding[label] = []
        for description in all_labeldescriptions_to_word_embedding[label]:
            labelname_to_word_embedding[label].append(all_labeldescriptions_to_word_embedding[label][description])
        labelname_to_word_embedding[label] = torch.stack(labelname_to_word_embedding[label], axis = 0)
    #---------------------LABEL NAME -> LABEL EMBEDDING ----------------------------#

    
    #------------------------ CLASS ID -> LABEL EMBEDDING --------------------------#
    # hyperparameters is a dictionary with many keys, one of them being labelname_to_word_embedding
    # This will be a dictionary with classid as key and its corresponding word embedding as value
    # labelname_to_classid_mapping[dataset][labelname] extracts the classid corresponding to a labelname and assigns the classid as a key of the dictionary hyperparameter['labelname_to_word_embedding']
    # labelname_to_word_embedding[labelname] extracts the word embedding corresponding to the labelname
    hyperparameters['classid_to_word_embedding'] = {}
    for labelname in labelname_to_word_embedding:
        hyperparameters['classid_to_word_embedding'][labelname_to_classid_mapping[dataset][labelname]] = labelname_to_word_embedding[labelname]
    #------------------------ CLASS ID -> LABEL EMBEDDING --------------------------#

    
    #------------------------ SENSOR EMBEDDING --------------------------#
    sensor_description_to_word_embedding_path = base_path + 'sensor_description_imagebind_embedding.pickle'
    with open(sensor_description_to_word_embedding_path, 'rb') as f:
        # sensordescription_to_word_embedding is a dictionary with dataset name as the keys
        # Each dataset key is also a dictionary with sensor names as the keys and their corresponding descriptions' word embedding as values
        sensor_description_to_word_embedding = pickle.load(f)[dataset]

    sensor_code_path = base_path + 'sensor_code.json'
    with open(sensor_code_path, 'r') as openfile:
        # Reading from json file
        sensor_code_list = json.load(openfile)[dataset]

    hyperparameters['sensor_description_to_word_embedding'] = []
    for sensor_code in sensor_code_list:
        hyperparameters['sensor_description_to_word_embedding'].append(sensor_description_to_word_embedding[sensor_code])
    hyperparameters['sensor_description_to_word_embedding'] = torch.stack(hyperparameters['sensor_description_to_word_embedding'], axis = 0)
    #------------------------ SENSOR EMBEDDING --------------------------#


    hyperparameters['classid_to_labelname_mapping'] = classid_to_labelname_mapping[dataset]
    hyperparameters['labelname_to_classid_mapping'] = labelname_to_classid_mapping[dataset]
    hyperparameters['folds_classid_to_labelname_mapping'] = folds_classid_to_labelname_mapping[dataset][fold]
    
    if fold == 'fold0': # fold0 is the regular time series classification case, all classes are both in training and test sets
        hyperparameters['num_train_classes'] = hyperparameters['nclasses']
        hyperparameters['num_test_classes'] = hyperparameters['nclasses']
    else:
        # number of training classes = total number of classes - number of classes specified in the fold
        hyperparameters['num_train_classes'] = hyperparameters['nclasses'] - len(hyperparameters['folds_classid_to_labelname_mapping'])
        if zsl:
            # zsl is defined as the test set will only contain the unseen classes NOT the ones that are present in training. 
            # So the number of test classes will be the number of classes specified in the fold
            hyperparameters['num_test_classes'] = len(hyperparameters['folds_classid_to_labelname_mapping'])
        else:
            # gzsl is harder than zsl. It is defined as the test set will contain both seen and unseen classes. 
            # So the number of test classes will be the total number of classes present in the dataset
            hyperparameters['num_test_classes'] = hyperparameters['nclasses']

    return hyperparameters



def data_loader_zsl(data_path, prop): 
    X_train = np.load(os.path.join(data_path + 'X_train.npy'), allow_pickle = True).astype(np.float)
    X_test = np.load(os.path.join(data_path + 'X_test.npy'), allow_pickle = True).astype(np.float)
    y_train = np.load(os.path.join(data_path + 'y_train.npy'), allow_pickle = True)
    y_test = np.load(os.path.join(data_path + 'y_test.npy'), allow_pickle = True)

    X = np.concatenate((X_train, X_test), axis = 0)
    y = np.concatenate((y_train, y_test), axis = 0)
    
    # select instances that contains permissible classes for the given fold
    # then filter X_train and y_train accordingly
    all_classes = set(list(range(prop['nclasses'])))
    test_classes = {int(key) for key in prop['folds_classid_to_labelname_mapping']}
    train_classes = all_classes - test_classes
    
    train_indices = []
    for class_id in train_classes:
        train_indices.extend(np.where(y == class_id))
    train_indices = np.concatenate(train_indices)
    np.random.shuffle(train_indices)
    X_train, y_train = X[train_indices], y[train_indices]

    test_indices = []
    for class_id in test_classes:
        test_indices.extend(np.where(y == class_id))
    test_indices = np.concatenate(test_indices)
    np.random.shuffle(test_indices)
    X_test, y_test = X[test_indices], y[test_indices]

    # ZSL, fold0 -> wrong
    # ZSL, fold1 / fold2 / fold3 -> normal ZSL

    assert prop['num_train_classes'] == len(set(y_train))
    assert prop['num_test_classes'] == len(set(y_test))

    # 2 cases
    # 1. ZSL, fold0 -> INVALID COMBINATION -> all classes in training, none in test set
    # 2. ZSL, fold1, fold2, fold3 -> some classes in training, remaining classes in test set

    return X_train, y_train, X_test, y_test



def data_loader_gzsl(data_path, prop): 
    X_train = np.load(os.path.join(data_path + 'X_train.npy'), allow_pickle = True).astype(np.float)
    X_test = np.load(os.path.join(data_path + 'X_test.npy'), allow_pickle = True).astype(np.float)
    y_train = np.load(os.path.join(data_path + 'y_train.npy'), allow_pickle = True)
    y_test = np.load(os.path.join(data_path + 'y_test.npy'), allow_pickle = True)

    # select instances from y_train that contains permissible classes for the given fold
    # then filter X_train and y_train accordingly
    train_class_ids = set(y_test) - {int(key) for key in prop['folds_classid_to_labelname_mapping']}
    train_indices = []
    for class_id in train_class_ids:
        train_indices.extend(np.where(y_train == class_id))
    train_indices = np.concatenate(train_indices)
    np.random.shuffle(train_indices)
    X_train, y_train = X_train[train_indices], y_train[train_indices]

    # GZSL, fold0 -> normal time series classification
    # GZSL, fold1 / fold2 / fold3 -> normal GZSL

    # if gzsl, then the test set remains same across all folds which is basically the entire test set containing all classes
    assert prop['num_train_classes'] == len(set(y_train))
    assert prop['num_test_classes'] == len(set(y_test))

    # 2 cases
    # 1. GZSL, fold0 -> all classes in training and test sets
    # 2. GZSL, fold1, fold2, fold3 -> some classes in training, all classes in test set

    return X_train, y_train, X_test, y_test



def make_perfect_batch(X, num_inst, num_samples):
    extension = np.zeros((num_samples - num_inst, X.shape[1], X.shape[2]))
    X = np.concatenate((X, extension), axis = 0)
    return X



def mean_standardize_fit(X):
    m1 = np.mean(X, axis = 1)
    mean = np.mean(m1, axis = 0)
    
    s1 = np.std(X, axis = 1)
    std = np.mean(s1, axis = 0)
    
    return mean, std



def mean_standardize_transform(X, mean, std):
    return (X - mean) / std



def preprocess(batch_size, X_tr, y_tr, X_te, y_te):
    
    X_train, y_train = shuffle(X_tr, y_tr)
    X_test, y_test = X_te, y_te

    num_train_inst, num_test_inst = X_train.shape[0], X_test.shape[0]
    num_train_samples = math.ceil(num_train_inst / batch_size) * batch_size
    num_test_samples = math.ceil(num_test_inst / batch_size) * batch_size
    
    X_train = make_perfect_batch(X_train, num_train_inst, num_train_samples)
    X_test = make_perfect_batch(X_test, num_test_inst, num_test_samples)

    X_train = torch.as_tensor(X_train).float()
    X_test = torch.as_tensor(X_test).float()
    y_train = torch.as_tensor(y_train)
    y_test = torch.as_tensor(y_test)
    
    return X_train, y_train, X_test, y_test



# Function to initialize training with different optimizers and different learning rates for different parts of the model
def initialize_training(prop):
    model = architecture.GZSLModel(prop['stage'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'], prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nlayers'], \
                                    prop['pooling'], prop['classid_to_word_embedding'], prop['sensor_description_to_word_embedding'], dropout = prop['dropout']).to(prop['device'])

    best_model = architecture.GZSLModel(prop['stage'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'], prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nlayers'], \
                                    prop['pooling'], prop['classid_to_word_embedding'], prop['sensor_description_to_word_embedding'], dropout = prop['dropout']).to(prop['device'])

    criterion = torch.nn.CrossEntropyLoss()
    
    parameters_time_series = list(model.trunk_net.parameters()) + list(model.transformer_encoder.parameters()) + list(model.batch_norm.parameters()) + list(model.cls_emb.parameters()) + list(model.downsample_embedding_to_input_net.parameters()) + list(model.sequence_representation_net.parameters()) + list(model.imu_to_shared_net.parameters())
    parameters_word_embedding = list(model.text_to_shared_net.parameters())

    optimizer_time_series = torch.optim.Adam(parameters_time_series, lr = prop['lr'])
    best_optimizer_time_series = torch.optim.Adam(parameters_time_series, lr = prop['lr'])

    optimizer_word_embedding = torch.optim.Adam(parameters_word_embedding, lr = prop['lr'] / prop['batch'])
    best_optimizer_word_embedding = torch.optim.Adam(parameters_word_embedding, lr = prop['lr'] / prop['batch'])

    return model, optimizer_time_series, optimizer_word_embedding, criterion, best_model, best_optimizer_time_series, best_optimizer_word_embedding



def compute_stage1_loss(model, device, criterion, batched_input):
    
    model.train()

    batch_size, seq_len, input_size = batched_input.shape # batch, seq, input

    cos_sim_imu_to_sensor = model(torch.as_tensor(batched_input, device = device)) # batch, input, input
    cos_sim_sensor_to_imu = cos_sim_imu_to_sensor.permute(0, 2, 1) # batch, input, input

    target = torch.as_tensor(list(range(input_size)), device = device) # input
    batched_target = target.repeat(batch_size) # batch * input = input

    cos_sim_imu_to_sensor = cos_sim_imu_to_sensor.reshape(batch_size * input_size, input_size) # batch * input, input = batch, input, input
    cos_sim_sensor_to_imu = cos_sim_sensor_to_imu.reshape(batch_size * input_size, input_size) # batch * input, input = batch, input, input

    loss_imu_to_sensor = criterion(cos_sim_imu_to_sensor, batched_target)
    loss_sensor_to_imu = criterion(cos_sim_sensor_to_imu, batched_target)

    correct_num = (torch.argmax(cos_sim_imu_to_sensor, 1) == batched_target).sum().detach().cpu().item()

    return loss_imu_to_sensor, loss_sensor_to_imu, 0.5 * (loss_imu_to_sensor + loss_sensor_to_imu), correct_num



def compute_stage2_loss(nclasses, model, device, criterion, y_train, batched_input, num_inst, start):
    model.train()
    out, attn = model(torch.as_tensor(batched_input, device = device))
    out = out.view(-1, nclasses)
    loss = criterion(out[ : num_inst], torch.as_tensor(y_train[start : start + num_inst], device = device)) # dtype = torch.long
    return attn, loss



def stage1_train_per_epoch(model, criterion, optimizer_time_series, optimizer_word_embedding, X_train, prop):

    model.train() # Turn on the train mode
    num_batches = math.ceil(X_train.shape[0] / prop['batch'])

    total_imu_to_sensor_loss, total_sensor_to_imu_loss, total_loss, total_correct_num = 0, 0, 0, 0
    for i in range(num_batches - 1):
        start = int(i * prop['batch'])
        end = int((i + 1) * prop['batch'])
        
        optimizer_time_series.zero_grad()
        optimizer_word_embedding.zero_grad()

        batched_input = X_train[start : end]
        
        # average loss per instance in a batch
        loss_imu_to_sensor, loss_sensor_to_imu, average_loss, correct_num = compute_stage1_loss(model, prop['device'], criterion, batched_input)
        
        total_imu_to_sensor_loss += loss_imu_to_sensor.item() * X_train.shape[2] * prop['batch']  # total loss in a batch
        total_sensor_to_imu_loss += loss_sensor_to_imu.item() * X_train.shape[2] * prop['batch']  # total loss in a batch
        total_loss += average_loss * X_train.shape[2] * prop['batch']  # total loss in a batch
        total_correct_num += correct_num

        average_loss.backward()

        optimizer_time_series.step()
        optimizer_word_embedding.step()
    
    denom = X_train.shape[2] * prop['batch'] * (num_batches - 1)

    # average loss across a single instance in the dataset
    return total_imu_to_sensor_loss / denom, total_sensor_to_imu_loss / denom, total_loss / denom, total_correct_num / denom



def stage2_train_per_epoch(model, criterion, optimizer_time_series, optimizer_word_embedding, X_train, y_train, prop):
    
    model.train() # Turn on the train mode
    num_batches = math.ceil(X_train.shape[0] / prop['batch'])
    
    total_loss = 0.0
    for i in range(num_batches):
        start = int(i * prop['batch'])
        end = int((i + 1) * prop['batch'])
        num_inst = y_train[start : end].shape[0]
        
        optimizer_time_series.zero_grad()
        optimizer_word_embedding.zero_grad()

        batched_input = X_train[start : end]
        
        # average loss per instance in a batch
        attn, loss = compute_stage2_loss(prop['nclasses'], model, prop['device'], criterion, y_train, batched_input, num_inst, start)
        total_loss += loss.item() * num_inst # total loss in a batch

        loss.backward()

        optimizer_time_series.step()
        optimizer_word_embedding.step()
    
    return total_loss / y_train.shape[0] # average loss across a single instance in the dataset



def evaluate(y_pred, y, criterion, prop):
    results = []

    if prop['zsl']:
        # for zsl, the test set will contain the unseen classes only
        test_keys = [int(key) for key in prop['folds_classid_to_labelname_mapping']]
    else:
        # for gzsl or regular clasification, the test set will contain both seen and unseen classes
        test_keys = [int(key) for key in range(prop['nclasses'])]
    sorted(test_keys)

    # extract the label names corresponding to the test set classes
    target_names = [prop['classid_to_labelname_mapping'][str(key)] for key in test_keys]

    loss = criterion(y_pred.view(-1, prop['nclasses']), torch.as_tensor(y, device = prop['device'])).item()
    
    # pred: (batch, nclasses), nclasses -> total number of classes in the dataset. original_target: batch
    pred, original_target = y_pred.cpu().data.numpy(), y.cpu().data.numpy()
    
    # pred: (batch, num_test_classes). For regular classification or gzsl, it doesn't make a difference because nclasses = num_test_classes
    # but for zsl, we only consider the unseen classes. So we need to extract the columns corresponding to the test classes
    pred = pred[ : , test_keys]

    # pred will be in range 0 to num_test_classes - 1. but original target may be like 8, 10, 13 because they are test set labels.
    # need to transform 8 -> 0, 10 -> 1, 13 -> 2
    # transformed_target: batch
    map = {k : i for i, k in enumerate(sorted(list(set(original_target))))}
    transformed_target = np.array([map[t] for t in original_target])
    
    # pred: batch, original_target: batch, transformed_target: batch
    pred = np.argmax(pred, axis = 1)
    acc = accuracy_score(transformed_target, pred)
    prec =  precision_score(transformed_target, pred, average = prop['avg'])
    rec = recall_score(transformed_target, pred, average = prop['avg'])
    f1 = f1_score(transformed_target, pred, average = prop['avg'])
    results.extend([loss, acc, prec, rec, f1])
    matrix = confusion_matrix(transformed_target, pred)
    acc_per_class = matrix.diagonal() / matrix.sum(axis = 1)

    
    if prop['zsl']:
        
        unseen_acc = np.mean(np.array(acc_per_class))
        seen_acc = 0
        avg_acc = unseen_acc
        
    else:
        
        if prop['fold'] == 'fold0':

            unseen_acc = 0
            seen_acc = np.mean(np.array(acc_per_class))
            avg_acc = seen_acc

        else:

            unseen_class_ids = set([int(key) for key in prop['folds_classid_to_labelname_mapping']])
            seen_class_ids = set(list(range(prop['nclasses']))) - unseen_class_ids
            
            unseen_acc = np.mean(np.array([acc_per_class[i] for i in unseen_class_ids]))
            seen_acc = np.mean(np.array([acc_per_class[i] for i in seen_class_ids]))

            avg_acc = (unseen_acc * len(unseen_class_ids) + seen_acc * len(seen_class_ids)) / (len(unseen_class_ids) + len(seen_class_ids))

    harmonic_mean = (2 * unseen_acc * seen_acc) / (unseen_acc + seen_acc)

    return results, classification_report(transformed_target, pred, target_names = target_names, digits = 4), acc_per_class, [unseen_acc, seen_acc, avg_acc, harmonic_mean]



def test(model, criterion, X, y, prop):
    model.eval() # Turn on the evaluation mode :v
    num_batches = math.ceil(X.shape[0] / prop['batch'])
    
    output_arr = []
    with torch.no_grad():
        for i in range(num_batches):
            start = int(i * prop['batch'])
            end = int((i + 1) * prop['batch'])
            num_inst = y[start : end].shape[0]
            
            out = model(torch.as_tensor(X[start : end], device = prop['device']))[0]
            output_arr.append(out[ : num_inst])

    return evaluate(torch.cat(output_arr, 0), y, criterion, prop)



def stage1_training(model, optimizer_time_series, optimizer_word_embedding, criterion, best_model, best_optimizer_time_series, best_optimizer_word_embedding, X_tr, prop):

    results_dir = prop['base_path'] + prop['expt_name'] + '/results/'
    if not os.path.exists(results_dir):
        # Create the directory
        os.makedirs(results_dir)

    model_dir = prop['base_path'] + prop['expt_name'] + '/models/'
    if not os.path.exists(model_dir):
        # Create the directory
        os.makedirs(model_dir)

    imu_to_sensor_loss_arr, sensor_to_imu_loss_arr, average_loss_arr, acc_arr, min_loss = [], [], [], [], math.inf

    for epoch in range(1, prop['epochs'] + 1):
        
        X_train = shuffle(X_tr)
        num_train_inst = X_train.shape[0]
        num_train_samples = math.ceil(num_train_inst / prop['batch']) * prop['batch']
        X_train = make_perfect_batch(X_train, num_train_inst, num_train_samples)
        X_train = torch.as_tensor(X_train).float()

        imu_to_sensor_loss, sensor_to_imu_loss, average_loss, acc = stage1_train_per_epoch(model, criterion, optimizer_time_series, optimizer_word_embedding, X_train, prop)
        
        imu_to_sensor_loss_arr.append(imu_to_sensor_loss)
        sensor_to_imu_loss_arr.append(sensor_to_imu_loss)
        average_loss_arr.append(average_loss)
        acc_arr.append(acc)
        
        if average_loss < min_loss:
            min_loss = average_loss
            best_model.load_state_dict(model.state_dict())
            best_optimizer_time_series.load_state_dict(optimizer_time_series.state_dict())
            best_optimizer_word_embedding.load_state_dict(optimizer_word_embedding.state_dict())

        print(prop['dataset'] + ', ' + str(prop['zsl']) + ', ' + prop['fold'] + ', GPU ' + str(prop['device']) + ', Epoch: ' + str(epoch) + ', I->S Loss: ' + str(imu_to_sensor_loss) + ', S->I Loss: ' + str(sensor_to_imu_loss) + ', Avg Loss: ' + str(average_loss) + ', Acc: ' + str(acc))
    
    model_file = model_dir + prop['dataset'] + '_' + str(prop['zsl']) + '_' + str(prop['fold']) + '_stage1_model_optimizer.pth'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'optimizer_time_series_state_dict': best_optimizer_time_series.state_dict(),
        'optimizer_word_embedding_state_dict': best_optimizer_word_embedding.state_dict(),
    }, model_file)

    results_file = results_dir + prop['dataset'] + '_' + str(prop['zsl']) + '_' + str(prop['fold']) + '_stage1_results.txt'
    with open(results_file, 'w') as file:
        for lst in [imu_to_sensor_loss_arr, sensor_to_imu_loss_arr, average_loss_arr, acc_arr]:
            line = ' '.join(map(str, lst))  # Convert each number to a string and join with spaces
            file.write(line + '\n')  # Write each list to a new line

    # Reading lists from a text file
    '''
    with open(results_file, 'r') as file:
        lists = []
        for line in file:
            number_list = list(map(float, line.split()))  # Convert each string back to a float
            lists.append(number_list)

    # Accessing the lists
    imu_to_sensor_loss_arr, sensor_to_imu_loss_arr, average_loss_arr, acc_arr = lists
    '''



def stage2_train_and_test(model, optimizer_time_series, optimizer_word_embedding, criterion, best_model, best_optimizer_time_series, best_optimizer_word_embedding, X_tr, y_tr, X_te, y_te, prop):
    
    results_dir = prop['base_path'] + prop['expt_name'] + '/results/'
    if not os.path.exists(results_dir):
        # Create the directory
        os.makedirs(results_dir)

    model_dir = prop['base_path'] + prop['expt_name'] + '/models/'
    if not os.path.exists(model_dir):
        # Create the directory
        os.makedirs(model_dir)

    results_file = open(results_dir + prop['dataset'] + '_' + str(prop['zsl']) + '_' + str(prop['fold']) + '_stage2_' + str(prop['pretrained']) + '_results', 'w')
    loss_arr, min_loss = [], math.inf
    global_test_metrics, global_acc_per_class, global_acc_metrics = [], [], []

    for epoch in range(1, prop['epochs'] + 1):
        
        X_train, y_train, X_test, y_test = preprocess(prop['batch'], X_tr, y_tr, X_te, y_te)
        loss = stage2_train_per_epoch(model, criterion, optimizer_time_series, optimizer_word_embedding, X_train, y_train, prop)
        loss_arr.append(loss)

        # save model and optimizer for lowest training loss on the end task
        if loss < min_loss:
            min_loss = loss
            best_model.load_state_dict(model.state_dict())
            best_optimizer_time_series.load_state_dict(optimizer_time_series.state_dict())
            best_optimizer_word_embedding.load_state_dict(optimizer_word_embedding.state_dict())

        print(prop['dataset'] + ', ' + str(prop['zsl']) + ', ' + prop['fold'] + ', GPU ' + str(prop['device']) + ', Epoch: ' + str(epoch) + ', Loss: ' + str(loss))
    print('Training complete...')
    print()

    model_file = model_dir + prop['dataset'] + '_' + str(prop['zsl']) + '_' + str(prop['fold']) + '_stage2_' + str(prop['pretrained']) + '_model_optimizer.pth'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'optimizer_time_series_state_dict': best_optimizer_time_series.state_dict(),
        'optimizer_word_embedding_state_dict': best_optimizer_word_embedding.state_dict(),
    }, model_file)



    print('Testing...')
    # Saved best model state at the lowest training loss is evaluated on the official test set
    test_metrics, report, acc_per_class, acc_metrics = test(best_model, criterion, X_test, y_test, prop) 

    print(test_metrics)
    print(report)
    print([key for key in prop['folds_classid_to_labelname_mapping']])
    print(acc_per_class)
    print(acc_metrics)

    reported_results_file = results_dir + prop['dataset'] + '_' + str(prop['zsl']) + '_' + str(prop['fold']) + '_stage2_' + str(prop['pretrained']) + '_reported_results.txt'
    with open(reported_results_file, 'w') as file:
        for lst in [loss_arr, test_metrics, acc_per_class, acc_metrics]:
            line = ' '.join(map(str, lst))  # Convert each number to a string and join with spaces
            file.write(line + '\n')  # Write each list to a new line

    print('Accuracy: ' + str(test_metrics[1]) + ', F1: ' + str(test_metrics[-1]))

    # Reading lists from a text file
    '''
    with open(reported_results_file, 'r') as file:
        lists = []
        for line in file:
            number_list = list(map(float, line.split()))  # Convert each string back to a float
            lists.append(number_list)

    # Accessing the lists
    loss_arr, global_test_metrics, global_acc_per_class, global_acc_metrics = lists
    '''