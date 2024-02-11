


from tkinter import *
from tkinter import ttk
from copy import deepcopy
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

warnings.filterwarnings("ignore")



data = pd.read_csv(r"C:\Users\Hazem\Desktop\4th\deep learning\tasks\task2\Dry_Bean_Dataset.xlsx")




# select_hidden = hidden_var
# select_eta = eta_var
# select_epoch = epoch_var
# select_neurons = neurons_var
# select_bias = bias_var
# select_activation = activation_var

data = data.fillna(data.mean())

X = data.drop("Class", axis=1)
y = data["Class"]

X = (X - X.min()) / (X.max() - X.min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

class Backpropagation:

    def _init_(self, select_hidden_layers, n_output_classes, learning_rate, select_epoch, select_neurons,
                 select_activation, select_bias):

        self.n_hidden = select_hidden_layers
        if type(select_hidden_layers) == int:
            self.n_hidden = [select_hidden_layers for _ in range(select_hidden_layers)]
        self.weight_init_const = 1
        self.learning_rate = learning_rate
        self.n_epoch = select_epoch
        self.select_bias = select_bias
        self.n_outputs = 1
        self.select_neurons = select_neurons
        self.select_activation = select_activation

    def weight_init(self):
        self.weights = []
        for layer, n_neurons in enumerate(self.n_hidden):
            if layer == 0:
                layer_weights = np.random.randn(self.n_feat, self.n_hidden[layer]) * self.weight_init_const
            else:
                layer_weights = np.random.randn(self.n_hidden[layer - 1], self.n_hidden[layer]) * self.weight_init_const

            self.weights.append(layer_weights)

        layer_weights = np.random.randn(self.n_hidden[-1], self.n_outputs) * self.weight_init_const
        self.weights.append(layer_weights)

    def calc_activation(self, net):
        if self.select_activation == "Sigmoid Function":
            return 1 / (1 + np.exp(-net))
        elif self.select_activation == "tanh":
            return (1 - np.exp(-net)) / (1 + np.exp(-net))

    def forward(self, X):
        self.layers = [X.reshape(-1,)]
        for Wi in self.weights:
            x = np.dot(self.layers[-1], Wi)
            self.layers.append(self.calc_activation(x))
        return self.layers[-1]

    def backward(self, X, y, y_pred):
        deltas = [y - y_pred]
        for i in reversed(range(len(self.weights))):
            Wi = self.weights[i]
            deltas.append(np.dot(deltas[-1], Wi.T) * (self.layers[i]))
        self.update_weights(deltas)

    def update_weights(self, deltas):
        self.weights = [Wi - self.learning_rate * np.dot(self.layers[i].T, delta)
                        for i, Wi, delta in zip(range(len(self.weights)), self.weights, reversed(deltas))]

    def train(self, x_train, y_train, use_pretrained_weights=False):
        if self.select_bias:
            x_train = deepcopy(x_train)
            x_train = np.c_[np.ones(x_train.shape[0]), x_train]

        self.n_feat = x_train.shape[1]

        if not use_pretrained_weights:
            self.weight_init()

        for inp in range(len(x_train)):
            pred = self.forward(x_train[inp])
            self.backward(x_train[inp], y_train[inp], pred)

    def predict(self, X_test):
        if self.select_bias:
            X_test = deepcopy(X_test)
            X_test = np.c_[np.ones(X_test.shape[0]), X_test]

        for _, weight in enumerate(self.weights):
            net = np.dot(X_test, weight)
            X_test = self.calc_activation(net)

        return np.argmax(X_test, axis=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def calculate_confusion_matrix(actual, predicted):
    tp = np.sum((actual == 1) & (predicted == 1))
    tn = np.sum((actual == 0) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    return tp, tn, fp, fn


def train_model():
    select_hidden = int(hidden_var.get())
    select_eta = float(eta_var.get())
    select_epoch = int(epoch_var.get())
    select_neurons = int(neurons_var.get())
    select_bias = bool(bias_var.get())
    select_activation = activation_var.get()

    # data = pd.read_excel('Dry_Bean_Dataset.xlsx')
    # data = data.fillna(data.mean())
    # X = data.drop("Class", axis=1)
    # y = data["Class"]
    # X = (X - X.min()) / (X.max() - X.min())
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)
    #
    # label_encoder = preprocessing.LabelEncoder()
    # y_train = label_encoder.fit_transform(y_train)
    # y_test = label_encoder.fit_transform(y_test)

    model = Backpropagation(select_hidden, 1, select_eta, select_epoch, select_neurons, select_activation, select_bias)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    tp, tn, fp, fn = calculate_confusion_matrix(y_test, y_pred)

    # result_text.delete(1.0, END)  # Clear previous results
    # result_text.insert(END, f"Confusion Matrix:\n"
    #                         f"True Positive: {tp}\n"
    #                         f"True Negative: {tn}\n"
    #                         f"False Positive: {fp}\n"
    #                         f"False Negative: {fn}\n\n")

    input_features = np.array([[0.885487133728734, 509.410076766318, 0.885487133728734, 509.410076766318, 509.410076766318]])
    output_class = model.predict(input_features)
    result_text.insert(END, f"Class ID: {output_class}\n\n")

    train_accuracy = model.accuracy(X_train, y_train)
    test_accuracy = model.accuracy(X_test, y_test)

    result_text.insert(END, f"Train Accuracy: {train_accuracy}\n"
                            f"Test Accuracy: {test_accuracy}")


root = Tk()
root.title("Neural Network GUI")

Label(root, text="Hidden Layers:").grid(row=0, column=0, padx=10, pady=5)
hidden_var = StringVar()
Entry(root, textvariable=hidden_var).grid(row=0, column=1)

Label(root, text="Learning Rate (eta):").grid(row=1, column=0, padx=10, pady=5)
eta_var = StringVar()
Entry(root, textvariable=eta_var).grid(row=1, column=1)

Label(root, text="Epochs:").grid(row=2, column=0, padx=10, pady=5)
epoch_var = StringVar()
Entry(root, textvariable=epoch_var).grid(row=2, column=1)

Label(root, text="Neurons per Layer:").grid(row=3, column=0, padx=10, pady=5)
neurons_var = StringVar()
Entry(root, textvariable=neurons_var).grid(row=3, column=1)

Label(root, text="Use Bias:").grid(row=4, column=0, padx=10, pady=5)
bias_var = BooleanVar()
Checkbutton(root, variable=bias_var).grid(row=4, column=1)

Label(root, text="Activation Function:").grid(row=5, column=0, padx=10, pady=5)
activation_var = StringVar()
activation_options = ttk.Combobox(root, textvariable=activation_var, values=["Sigmoid Function", "tanh"])
activation_options.grid(row=5, column=1)

train_button = Button(root, text="Train Model", command=train_model)
train_button.grid(row=6, column=0, columnspan=2, pady=10)

result_text = Text(root, height=10, width=50)
result_text.grid(row=7, column=0, columnspan=2, pady=10)

root.mainloop()


