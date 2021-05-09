import os
import sys
import numpy as np
import argparse
import yaml
from matplotlib import pyplot as plt
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)


def get_config_file():
    # Load the configuration file
    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--config", required=True, help="Path to where the config file resides")
    args = vars(ap.parse_args())

    config_path = args['config']
    conf = yaml.safe_load(open(config_path, 'r'))

    return conf


def read_frame_data(f_path, sample_type, gauss_pixel, channels=False):
    if sample_type not in f_path:
        f_path += sample_type

    #parameters_path = f_path.replace(sample_type, 'parameters.txt')

    samples_paths = get_files(f_path)
    frames, dataX, dataY = get_modeled_samples(samples_paths)
    #parameters = pd.read_csv(parameters_path, sep=' ')

    return frames, dataX, dataY

def get_modeled_samples(samples_paths):
    frame = []
    dataX = []
    dataY = []

    for p in samples_paths:
        sample = pd.read_csv(p)
        positions = sample.values.astype(np.float)
        #print(positions)
        frame.append(positions[:][0])
        #dataX.append(positions[1])
        #dataY.append(positions[-1])

    return np.array(frame), np.array(dataX), np.array(dataY)


def get_files(dir_path):
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        file_paths = [os.path.join(root, file) for file in files if file.endswith(".txt")]

    return file_paths

class Net(object):

    def __init__(self, net_type, **kwargs):
        self.net_type = net_type
        self.framework = kwargs['framework']
        print(self.framework)
        if 'model_file' in kwargs.keys():
            self.model_path = kwargs['model_file'][:kwargs['model_file'].rfind("/") + 1]
            if self.framework == "keras":
                self.model = load_model(kwargs['model_file'])
            else:
                self.model = tf.keras.models.load_model(kwargs['model_file'])
        else:
            if self.framework == "keras":
                self.model = Sequential()
            else:
                self.model = tf.keras.Sequential()

            self.dropout = kwargs['dropout']
            if self.dropout:
                self.drop_percentage = kwargs['drop_percentage']
            self.loss = kwargs['loss']
            self.activation = kwargs['activation']
            self.input_shape = kwargs['input_shape']
            self.output_shape = kwargs['output_shape']

    def train(self, n_epochs, batch_size, patience, root, data_train, data_val, batch_data, gauss_pixel, channels):
        utils.check_dirs(root)

        name = root + '/' + str(batch_size) + '_' + str(self.dropout) + '_' + self.activation + '_' + \
            self.loss + '_' + str(patience)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience)
        checkpoint = ModelCheckpoint(name + '.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
        print(name)

        print('Training model...')
        start_time = time()
        print("No batch data")
        model_history = self.model.fit(data_train[0], data_train[1], batch_size=batch_size,
                                       epochs=n_epochs, validation_data=data_val,
                                       callbacks=[early_stopping, checkpoint], verbose=1)
        end_time = time()

        print("End training")

        if len(model_history.epoch) < n_epochs:
            n_epochs = len(model_history.epoch)

        self.save_properties(patience, n_epochs, round(end_time-start_time, 2), name + '_properties')
        utils.save_history(model_history, name)

    def save_properties(self, patience, epochs, train_time, file_path):
        if self.framework == "keras":
            vis_utils.plot_model(self.model, file_path + '.png', show_shapes=True)
        else:
            tf.keras.utils.plot_model(self.model, file_path + '.png', show_shapes=True)

        with open(file_path + '.txt', 'w+') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write('\n\n-----------------------------------------------------------------------------------\n\n')
            f.write('Patience: ' + str(patience) + '\n')
            f.write('Epochs: ' + str(epochs) + '\n')
            f.write('Execution time: ' + str(train_time) + '\n')

class Lstm(Net):
    def __init__(self, **kwargs):
        Net.__init__(self, "lstm", **kwargs)
        if 'model_file' not in kwargs.keys():
            self.create_frame_complex_model()


    def create_frame_complex_model(self):
        print("Creating frame complex LSTM model")

        self.model.add(LSTM(70, return_sequences=True,  input_shape=self.input_shape))
        self.model.add(LSTM(40, return_sequences=True))
        self.model.add(LSTM(25, return_sequences=True))
        self.model.add(LSTM(15, return_sequences=False))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape))
        self.model.compile(loss=self.loss, optimizer='adam')


if __name__ == '__main__':
    conf = get_config_file()

    net_type = conf['net_type']
    activation = conf['activation']
    dropout = conf['dropout']['flag']
    drop_percentage = float(conf['dropout']['percentage'])
    n_epochs = conf['n_epochs']
    batch_size = conf['batch_size']
    patience = conf['patience']

    data_dir = conf['data_dir']
    data_type = data_dir.split('/')[5]
    func_type = data_dir.split('/')[6]

    root = conf['root'] + net_type.upper() + '/' + data_type + '/' + func_type
    print(root)
    version = conf['version']

    print('Puting the data into the right shape...')

    # data_type == 'Frames_dataset':
    print('Training with frames')
    data_model = conf['data_model']
    samples_dir = data_dir.split('/')[6]
    dim = (int(samples_dir.split('_')[-2]), int(samples_dir.split('_')[-1]))
    complexity = conf['complexity']

    # Load data
    channels = False
    print("Modeled images")
    batch_data = False
    loss = conf['modeled_frame_loss']
    gauss_pixel = False
    activation = conf['modeled_activation']
    dim = (int(samples_dir.split('_')[-2]), int(samples_dir.split('_')[-1]))
    filename = root + "_Modeled/" + complexity

    frames, trainX, trainY = read_frame_data(data_dir + 'train/', 'modeled_samples', gauss_pixel)
    print('FRAMES ---------------------------')
    print(frames)
    print('TRAIN X ---------------------------')
    print(trainX)
    print('TRAIN Y ---------------------------')
    print(trainY)
    #_, frames_val, valX, valY = frame_utils.read_frame_data(data_dir + 'val/', 'modeled_samples', gauss_pixel)
    #train_data = (frames, trainX, trainY)
    #print(train_data)
    #val_data = (valX, valY)
    #print(frames_val, val_data)

    # Model settings
    #in_dim = trainX.shape[1:]
    #out_dim = np.prod(in_dim[1:])
    #to_train_net = Net.Lstm(activation=activation, loss=loss, dropout=dropout,
                                #drop_percentage=drop_percentage, input_shape=in_dim, output_shape=out_dim,
                                #complexity=complexity, data_type="Frame", framework="keras")

    #print('Training')
    #to_train_net.train(n_epochs, batch_size, patience, filename, train_data, val_data, batch_data, gauss_pixel, channels)
