from src.mnist_loader import load_data_wrapper
from src.network import Network


def main():
    file = open("testfile.txt", "w")
    file.write("This is a test")
    file.close()

    training_data, validation_data, test_data = load_data_wrapper()



    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

if __name__ == '__main__':
   main()