import pandas as pd
import numpy as np
import os
from neural_net import EzNet

_op_table = {
    'train': {'train_data': None,
              'epochs': None,
              'test_data': None,
              'hlayers': None,
              'classes': None,
              'learn_rate': None,
              'outfile': None},
    'predict': {'infile': None,
                'outfile': None},
    'save': {'save_dir': None},
    'load': {'load_dir': None}
}

_net = 0


def print_intro():
    print('Welcome to EzNet. This is an easy to use feed forward neural network. You can train a '
          'neural network and then use it to predict values. You can also save the current neural '
          'network weight values into a directory of choice and recover them later. For more '
          'detailed instruction, refer to the user guide inside the docs directory.')


def print_table():
    """
    Prints the table of commands with their configurable options.
    :return: None
    """
    for key in _op_table.keys():
        print(key)
        for sub_key in _op_table[key]:
            print('\t--' + sub_key)


def fill_op_table(primary, options):
    """
    Uses the given options to populate the op table. The given options must correspond to the
    given op table options and be of correct data type.
    :param primary: The primary command to execute (train, predict, test, etc)
    :param options: The configurable options for the primary command
    :return: None
    """
    for item in options:
        key = item[2:item.index('=')]
        val = item[(item.index('=') + 1):]
        if primary == 'train':
            if key == 'train_data' or key == 'test_data' or key == 'outfile':
                _op_table[primary][key] = str(val)
            elif key == 'epochs' or key == 'classes':
                _op_table[primary][key] = int(val)
            elif key == 'hlayers':
                _op_table[primary][key] = [int(num) for num in val.split(sep=',')]
            else:  # if key == 'learn_rate'
                _op_table[primary][key] = float(val)
        elif primary == 'predict':
            _op_table[primary][key] = str(val)
        elif primary == 'save':
            _op_table[primary][key] = str(val)
        elif primary == 'load':
            _op_table[primary][key] = str(val)


def exec_cmd(key):
    """
    Executes the command corresponding to the given key. The key must correspond to one of the
    executable commands in the op table, and the options for the given command must be set prior
    to executing.
    :param key: The command that is to be executed (train, predict, test, etc)
    :return: None
    """
    global _net
    sub_table = _op_table[key]
    if key == 'train':
        mat_train = pd.read_csv(sub_table['train_data'], header=None).values
        x_train = mat_train[0:1000, 1:]
        y_train = mat_train[0:1000, 0]
        mat_test = pd.read_csv(sub_table['test_data'], header=None).values
        x_test = mat_test[:, 1:]
        y_test = mat_test[:, 0]
        _net = EzNet(np.size(x_train, axis=1), sub_table['hlayers'], sub_table['classes'])
        test_loss, train_loss, acc = _net.train(x_train, y_train, sub_table['learn_rate'],
                                                sub_table['epochs'], x_test, y_test)
        df = pd.DataFrame()
        df['test_loss'] = test_loss
        df['train_loss'] = train_loss
        df['accuracy'] = acc
        df.to_csv(sub_table['outfile'], sep=',')
    elif key == 'predict':
        mat_pred = pd.read_csv(sub_table['infile'], header=None).values
        x_pred = mat_pred[:, 1:]
        y_pred = mat_pred[:, 0]
        preds = _net.predict(x_pred)[0]
        df = pd.DataFrame()
        df['predictions'] = preds
        df['actual'] = y_pred
        df.to_csv(sub_table['outfile'], sep=',')
    elif key == 'save':
        i = 0
        if not os.path.exists(sub_table['save_dir']):
            os.makedirs(sub_table['save_dir'])
        for theta in _net._thetas:
            fname = sub_table['save_dir'] + '\\weights-layer-' + str(i) + '.csv'
            print('saving file: ', fname)
            np.savetxt(fname, theta, delimiter=',')
            i += 1
    elif key == 'load':
        theta_files = os.listdir(sub_table['load_dir'])
        sample = pd.read_csv(sub_table['load_dir'] + '\\weights-layer-0.csv', header=None).values
        _net = EzNet(np.size(sample, axis=1) - 1, None, 1)
        _net._thetas = []
        for i in range(0, len(theta_files)):
            fname = sub_table['load_dir'] + '\\weights-layer-' + str(i) + '.csv'
            _net._thetas.append(pd.read_csv(fname, header=None).values)


def clear_table():
    for primary in _op_table.keys():
        for sub in _op_table[primary].keys():
            _op_table[primary][sub] = None


def run():
    """
    In a loop, gets user input and parses it and then runs the corresponding commands. Prints the
    intro and the table at the very start.
    :return: None
    """
    print_intro()
    print_table()
    while True:
        inp = input('>>').split(sep=' ')
        fill_op_table(inp[0], inp[1:])
        exec_cmd(inp[0])
        clear_table()


if __name__ == '__main__':
    """
    Calls the run method
    """
    run()
