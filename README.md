# Neural-net
Easy to use feed forward neural network

Instructions:
to run, navigate to src and run the python script command_line_tools.py
to perform a specific task, enter: task_name --option_1=.. and so on so forth for all options
for example, to train:
Markup: * train --train_data=../my_train_data.csv epochs=50 test_data../my_test_data.csv hlayers=14,12 classes=6 learn_rate0.01 outfile=../results.csv
Note that ALL options for the required task must be filled in

Training options:
Markup: * train_data: a csv file containing training data. The first column of each line should be the classification, with the rest being features
        * epochs: the number of epochs to train for
        * test_data: a csv file containing data to test for. Test results are generated after each epoch to compare to training results at the same time. Same format as train data
        * hlayers: hidden layer configuration of network. List of numbers, no spaces. Each entry in the list contains the number of neurons the network contains in the respective hidden layer, going in order from closes to input to closes to output.
        * classes: number of possible classifications for the network
        * learn_rate: the learning rate to use during gradient descent

Predict options:
Markup: * infile: input file to predict with. Format should be the same as other csv files such as train_data and test_data
        * outifle: file in which the results of the predictions are stored

Save options:
Markup: * save_dir: directory in which to save the weights of the neural network as they currently are

Load options:
Markup: * load_dir directory from which to load saved neural network files
