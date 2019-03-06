import pickle

from src.mnist_loader import load_data_wrapper
from src.network import Network

FILEPATH = "trainedNetwork.pkl"

def main():

    net = None

    try:
        loadNetwork(net, FILEPATH)
        print("Network loaded with success.")

    except:
        print("Did not found a previous network trained. Will train one.")
        trainNetwork(net)
        # sample usage
        saveNetwork(net, FILEPATH)


def saveNetwork(obj, filepath):
    with open(filepath, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadNetwork(obj, filepath):
    with open(filepath, 'rb') as input:
        obj = pickle.load(input)

def trainNetwork(net):
    training_data, validation_data, test_data = load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

if __name__ == '__main__':
   main()