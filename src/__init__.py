import pickle

from src.mnist_loader import load_data_wrapper
from src.network import Network

FILEPATH = "trainedNetwork.pkl"

def main():

    net = None

    try:
        net = loadNetwork(FILEPATH)
        print("Network loaded with success.")

    except:
        print("Did not found a previous network trained. Will train one.")
        net = trainNetwork()
        # sample usage
        saveNetwork(net, FILEPATH)

    print("net: " , net)

    #this is giving wrong lenght data...
    #print("Result: ", net.getNumberFromImage("number-212.png"))
    #print("------")

    #print("Result (4):\n", net.getNumberFromImage("number-219.png"))
    print("------")
    #print("Result (7):\n", net.getNumberFromImage("number-45.png"))
    print("------")
    #print("Result: ", net.getNumberFromImage("eggs.png"))


def saveNetwork(obj, filepath):
    with open(filepath, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadNetwork(filepath):
    with open(filepath, 'rb') as input:
        obj = pickle.load(input)
        return obj

def trainNetwork():
    training_data, validation_data, test_data = load_data_wrapper()
    net = Network([784, 60, 30, 10])
    net.SGD(training_data, 50, 10, 3.0, test_data=test_data)
    return net

if __name__ == '__main__':
   main()