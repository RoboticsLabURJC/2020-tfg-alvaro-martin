''' TEST NEW '''


import sys
sys.path.insert(0, '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test')
#sys.path.insert(0, 'C:/Users/optiva/Desktop/TFG/2020-tfg-alvaro-martin/Generator & Train_Test/')

from Utils import utils, func_utils, vect_utils, frame_utils
from Network import Net

'''
def test(self, test_x, test_y, gap, data_type, dim):
    predict = self.model.predict(test_x)
    if data_type == "Functions_dataset":
        maximum = [np.max(np.abs(np.append(test_x[i], test_y[i]))) for i in range(len(test_x))]
        predict_values = predict
        real_values = test_y
        v_to_draw = predict_values
    elif data_type == "Vectors_dataset":
        predict_values, real_values, maximum = vect_utils.get_positions(predict, test_y)
        v_to_draw = predict_values
    else:
        raw = True
        if "modeled" in data_type:
            raw = False
        predict_values, real_values, maximum = frame_utils.get_positions(predict, test_y, dim, raw)

        if raw:
            v_to_draw = predict
        else:
            v_to_draw = predict_values

    error, x_error, y_error, relative_error = test_utils.calculate_error(real_values, predict_values, maximum)

    with open(self.model_path + 'error_result.txt', 'w') as file:
        for i in range(error.shape[0]):
            file.write("Processed sample " + str(i) + ": \n")
            file.write("Target position: " + str(real_values[i]) + "\n")
            file.write("Position: " + str(predict_values[i]) + "\n")
            file.write("Error: " + str(np.round(error[i], 2)) + " (" + str(np.round(relative_error[i], 2)) + "%)\n")
            file.write("--------------------------------------------------------------\n")




    # Calculate stats
    test_utils.get_error_stats(test_x, test_y, v_to_draw, gap, data_type, dim,
                               error, x_error, y_error, relative_error, self.model_path)

'''

if __name__ == '__main__':
    data_path = '/Users/Martin/Desktop/Generator_10/Frames_dataset/linear_point_255_fix_6000_80_120/linear_10_[None]_test'
    model_path = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test/Models/REC/Frames_dataset/linear_point_255_fix_6000_80_120_Modeled/convLSTM/10_False_tanh_mean_squared_error_10.h5'

    data_type = data_path.split('/')[6]
    net_type = model_path.split('/')[7]
    complexity = model_path.split('/')[10]

    print('\n')
    print("Dataset: " + data_path)
    print('\n')
    print("Model: " + model_path)
    print('\n')

    print("Evaluating with " + data_type + " a " + complexity + " " + net_type + " model")

    if "modeled" in model_path.lower():
        data_path = data_path + "/modeled_samples"
    else:
        data_path = data_path + "/raw_samples"
    print(data_path)
    print('\n')
    sample_type = data_path.split('/')[-1]
    print(sample_type)
    data_type = data_type + "_" + sample_type
    samples_dir = data_path.split('/')[6]
    dim = (int(samples_dir.split('_')[-2]), int(samples_dir.split('_')[-1]))

    if sample_type == "raw_samples":
        gauss_pixel = "gauss" in model_path.lower()
        print("Gauss:", gauss_pixel)
        if net_type == "NOREC":
            print('Puting the test data into the right shape...')
            parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, gauss_pixel)
            to_test_net = Net.Convolution2D(model_file=model_path, framework="keras")
        else:
            print('Puting the test data into the right shape...')
            parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, gauss_pixel, True)
            to_test_net = Net.ConvolutionLstm(model_file=model_path, framework="keras")
    else:
        parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, False)
        if net_type == "NOREC":
            print('Puting the test data into the right shape...')
            to_test_net = Net.Mlp(model_file=model_path, framework="tensorflow")
        else:
            print('Puting the test data into the right shape...')
            to_test_net = Net.Lstm(model_file=model_path, framework="tensorflow")

    gap = parameters.iloc[0]['gap']

    to_test_net.test(testX, testY, gap, data_type, dim)
