"""

TFM - main_gen.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "23/04/2018"

import sys
#sys.path.insert(0, '/home/docker/TFG/2020-tfg-alvaro-martin/Generator & Train_Test/Generator')
sys.path.insert(0, '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator')
sys.path.insert(0, '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test')


from Utils.utils import write_header, check_dirs, get_config_file
import Frames, Shapes


if __name__ == '__main__':

    conf = get_config_file()

    n_samples = int(conf['n_samples'])  # Number of samples to save in the data set
    n_points = int(conf['n_points'])  # Number of points used to make prediction
    gap = int(conf['gap'])  # Separation between last sample and sample to predict
    noise_flag = conf['noise']['flag']  # Introduce noise to the samples
    split_flag = conf['split']['flag']
    if noise_flag:
        mean = float(conf['noise']['mean'])
        stand_deviation = float(conf['noise']['stand_deviation'])
        noise_parameters = [mean, stand_deviation]
    else:
        noise_parameters = [None]

    to_generate = conf['to_generate']  # Type to generate
    data_dir = conf['root'] + "_" + str(gap)


    h = conf['height']
    w = conf['width']
    obj_shape = conf['object']
    obj_color = conf['obj_color']
    motion_type = conf['motion_type']
    dof = conf['dof']
    data_dir += "/Frames_dataset/" + motion_type + '_' + dof + '_' + str(n_samples)
    if obj_shape == 'point':
        shape = Shapes.Point(obj_color)

    data_dir = data_dir + "_" + str(h) + "_" + str(w)
    check_dirs(data_dir, True)

    for i in range(n_samples):
        if i % 100 == 0 or i == n_samples - 1:
            print(i)

        if motion_type == 'linear':
            if i == 0:
                header = 'x0 u_x y0 m n_points gap motion noise(mean, standard deviation)\n'
            sample = Frames.Linear(noise_parameters, n_points, gap, h, w, shape, dof)

        if split_flag:
            if i == 0:
                train_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                 + str(noise_parameters) + '_train'
                check_dirs(train_path + '/raw_samples')
                check_dirs(train_path + '/modeled_samples')

                test_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                + str(noise_parameters) + '_test'
                check_dirs(test_path + '/raw_samples')
                check_dirs(test_path + '/modeled_samples')

                val_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                               + str(noise_parameters) + '_val'
                check_dirs(val_path + '/raw_samples')
                check_dirs(val_path + '/modeled_samples')

                write_header(train_path + '/parameters.txt', header)
                write_header(test_path + '/parameters.txt', header)
                write_header(val_path + '/parameters.txt', header)

                n_test = int(n_samples * float(conf['split']['fraction_test']))
                n_val = int(n_samples * float(conf['split']['fraction_validation']))
                n_train = n_samples - n_val - n_test

            if i < n_train:
                sample.save(train_path + '/raw_samples/sample' + str(i), train_path + '/parameters.txt',
                            train_path + '/modeled_samples/sample' + str(i) + '.txt')
            elif i < n_train + n_test:
                sample.save(test_path + '/raw_samples/sample' + str(i), test_path + '/parameters.txt',
                            test_path + '/modeled_samples/sample' + str(i) + '.txt')
            else:
                sample.save(val_path + '/raw_samples/sample' + str(i), val_path + '/parameters.txt',
                            val_path + '/modeled_samples/sample' + str(i) + '.txt')

        else:
            if i == 0:
                data_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                             + str(noise_parameters)
                check_dirs(data_path + '/samples')

                write_header(data_path + '/parameters.txt', header)

            sample.save(data_path + '/samples/sample' + str(i), data_path + '/parameters.txt')
