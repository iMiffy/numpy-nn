import numpy as np

# The first 3 functions in this file are from the Stanford cs231n course.
# In gradient checking, to get an approximate gradient for a parameter, we vary that parameter by a small amount (while keeping rest of parameters constant) and note the difference in the network loss. Dividing the difference in network loss by the amount we varied the parameter gives us an approximation for the gradient. We repeat this process for all the other parameters to obtain our numerical gradient. Note that gradient checking is a slow process (2 forward propagations per parameter) and should only be used to check your backpropagation!
# More links on gradient checking: http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/,  https://www.coursera.org/learn/machine-learning/lecture/Y3s6r/gradient-checking

def eval_numerical_gradient_inputs(layer, inputs, in_grads, h=1e-5):
    grads = np.zeros_like(inputs)
    it = np.nditer(inputs, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        oldval = inputs[idx]
        inputs[idx] = oldval + h
        pos = layer.forward(inputs).copy()
        inputs[idx] = oldval - h
        neg = layer.forward(inputs).copy()
        inputs[idx] = oldval

        grads[idx] = np.sum((pos - neg) * in_grads) / (2 * h)
        it.iternext()
    return grads


def eval_numerical_gradient_params(layer, inputs, in_grads, h=1e-5):
    w_grad = np.zeros_like(layer.weights)
    b_grad = np.zeros_like(layer.bias)

    w_it = np.nditer(w_grad, flags=['multi_index'], op_flags=['readwrite'])
    b_it = np.nditer(b_grad, flags=['multi_index'], op_flags=['readwrite'])

    while not w_it.finished:
        idx = w_it.multi_index

        oldval = layer.weights[idx]
        layer.weights[idx] = oldval + h
        pos = layer.forward(inputs).copy()
        layer.weights[idx] = oldval - h
        neg = layer.forward(inputs).copy()
        layer.weights[idx] = oldval

        w_grad[idx] = np.sum((pos - neg) * in_grads) / (2 * h)
        w_it.iternext()

    while not b_it.finished:
        idx = b_it.multi_index

        oldval = layer.bias[idx]
        layer.bias[idx] = oldval + h
        pos = layer.forward(inputs).copy()
        layer.bias[idx] = oldval - h
        neg = layer.forward(inputs).copy()
        layer.bias[idx] = oldval

        b_grad[idx] = np.sum((pos - neg) * in_grads) / (2 * h)
        b_it.iternext()

    return w_grad, b_grad


def eval_numerical_gradient_loss(loss, inputs, targets, h=1e-5):
    grads = np.zeros_like(inputs)
    it = np.nditer(inputs, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        oldval = inputs[idx]
        inputs[idx] = oldval + h
        pos = loss.forward(inputs, targets)[0].copy()
        inputs[idx] = oldval - h
        neg = loss.forward(inputs, targets)[0].copy()
        inputs[idx] = oldval

        grads[idx] = np.sum((pos - neg)) / (2 * h)
        it.iternext()
    return grads


def check_grads(cacul_grads, numer_grads):
    precise = np.linalg.norm(cacul_grads-numer_grads) / \
        max(np.linalg.norm(cacul_grads), np.linalg.norm(numer_grads))
    return precise


def check_grads_layer(layer, inputs, in_grads):
    map_bool = {
        'True': 'correct',
        'False': 'wrong',
    }

    numer_grads = eval_numerical_gradient_inputs(layer, inputs, in_grads)
    cacul_grads = layer.backward(in_grads, inputs)
    inputs_result = check_grads(cacul_grads, numer_grads)
    # print('<1e-8 will be fine')
    print('Gradient to input:', map_bool[str(inputs_result < 1e-8)])
    if layer.trainable:
        w_grad, b_grad = eval_numerical_gradient_params(
            layer, inputs, in_grads)
        w_results = check_grads(layer.w_grad, w_grad)
        b_results = check_grads(layer.b_grad, b_grad)
        print('Gradient to weights: ', map_bool[str(w_results < 1e-8)])
        print('Gradient to bias: ', map_bool[str(b_results < 1e-8)])

def check_grads_layer_error(layer, inputs, in_grads):
    results = []
    numer_grads = eval_numerical_gradient_inputs(layer, inputs, in_grads)
    cacul_grads = layer.backward(in_grads, inputs)
    inputs_result = check_grads(cacul_grads, numer_grads)
    results.append(inputs_result)
    if layer.trainable:
        w_grad, b_grad = eval_numerical_gradient_params(
            layer, inputs, in_grads)
        w_results = check_grads(layer.w_grad, w_grad)
        b_results = check_grads(layer.b_grad, b_grad)
        results.append(w_results)
        results.append(b_results)
    return results


def check_grads_loss(layer, inputs, targets):
    map_bool = {
        'True': 'correct',
        'False': 'wrong',
    }
    numer_grads = eval_numerical_gradient_loss(layer, inputs, targets)
    cacul_grads = layer.backward(inputs, targets)
    inputs_result = check_grads(cacul_grads, numer_grads)
    print('Gradient to input:', map_bool[str(inputs_result < 1e-8)])
