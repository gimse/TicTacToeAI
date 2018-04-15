import numpy as np


class mlp:
    def __init__(self, num_nodes):
        # number of nodes in each layer (including input and output)
        self.num_nodes = num_nodes
        self.weights = make_weights(num_nodes)
        self.u_list = make_u(num_nodes)
        self.iterations_per_layer = 1;

        self.error_function=np.array([])
        self.error_iterations=np.array([])



    def calculate(self, u_in):
        u_in = fix_shape(u_in)

        self.u_list[0] = u_in
        for i in range(len(self.weights)):
            # Add the inputs that match the bias node
            u0 = add_bias_to_u(self.u_list[i])
            # multipli with weights
            u1 = np.dot(self.weights[i], u0)
            # Use rounding function
            self.u_list[i + 1] = soft_max(u1)
        return self.u_list[-1]

    def learn(self, u, t, eta, niterations,momentum):
        u = fix_shape(u)
        t = fix_shape(t)

        assert np.shape(u)[0] == self.num_nodes[0]
        assert np.shape(t)[0] == self.num_nodes[-1]

        for n in range(niterations):

            for weight_number in reversed(range(len(self.weights))):

                old_grad = np.zeros(np.shape(self.weights[weight_number]));
                for iteration in range(self.iterations_per_layer): #not in use
                    self.calculate(u)
                    grad = -self.find_gradient(t, weight_number)
                    grad += momentum * old_grad;
                    old_grad = grad
                    self.weights[weight_number] += eta * grad

            error = self.error_last_calculation(t)
            if (np.mod(n, 100) == 0):
                #print("Iteration: ", n, " Error: ", error)
                self.error_function=np.append(self.error_function,error)
                self.error_iterations=np.append(self.error_iterations,n)


    def find_gradient(self, t, weight_number):
        # The function fins the gradient to E with respect i weights with number 'weight_number'.
        # It is kind of hard to explain how it works without seeing the math.
        # It goes thought every possible path in the mlp. It tok me ages to make!!

        n = len(self.weights)
        assert weight_number < n, 'dont have that many weights'
        assert weight_number >= 0, 'dont have negative weights'
        # de/dU
        sum = self.u_list[n] - t  # Inititlize with the error dirvivative
        return self.rekursiv_gradient(sum, n, n, weight_number)

    def rekursiv_gradient(self, sum, n_start, n, n_end):

        u = self.u_list[n]
        w = self.weights[n - 1]

        u2 = dg_du(u);  # Derivative of softmax
        # Combine derivate of softmax and sum (sum is dE/dU in first case)
        sum = np.multiply(sum, u2)

        # When at the last iteration
        if n == n_end + 1:
            u_end = self.u_list[n - 1]
            u_end = add_bias_to_u(u_end)  # add one to fit bias

            ans = np.dot(sum, np.transpose(u_end))
            return ans

        # if not at last iteration
        w = w[:, 1:]  # remove the bias from the weights
        sum = np.dot(np.transpose(w), sum)  # combine sum and weights

        return self.rekursiv_gradient(sum, n_start, n - 1, n_end)

    def error_last_calculation(self, t):
        return 0.5 * np.sum(np.square(self.u_list[-1] - t))

    def confmat(self, inputs, targets):
        # Copy from book with modifications

        outputs = self.calculate(inputs)

        nclasses = np.shape(targets)[0]
        outputs = np.argmax(outputs, 0)
        targets = np.argmax(targets, 0)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ", np.trace(cm) / np.sum(cm))
        return np.trace(cm) / np.sum(cm)

def fix_shape(u):
    shape = np.shape(u)
    # if vector
    if len(shape) <= 1:
        u = np.reshape(u, (len(u), 1))
    return u


def add_bias_to_u(u):
    shape_u = np.shape(u)
    l = np.ones((shape_u[1]))
    return np.vstack((l, u))


def make_weights(num_nodes):
    weight_list = []
    for i in range(len(num_nodes) - 1):
        input_nodes = num_nodes[0]
        print('input', input_nodes)
        m = num_nodes[i + 1]
        n = num_nodes[i] + 1
        w = (2 * np.random.random((m, n)) - np.ones((m, n))) / np.sqrt(input_nodes)
        weight_list.append(w)
    return weight_list


def make_u(num_nodes):
    u_list = []
    for i in range(len(num_nodes)):
        m = num_nodes[i]
        u = np.zeros(m)
        u_list.append(u)
    return u_list


def soft_max(h):
    return 1 / (np.ones(np.shape(h)) + np.exp(-h))


def dg_du(u):
    return np.multiply(np.ones(np.shape(u)) - u, u)