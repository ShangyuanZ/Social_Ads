from problem import *

if __name__ == "__main__":
    path = '..'
    X_train, y_train = get_train_data(path)
    X_test, y_test = get_test_data(path)
    print ("train", X_train.shape, y_train.shape)
    print ("test", X_test.shape, y_test.shape)
