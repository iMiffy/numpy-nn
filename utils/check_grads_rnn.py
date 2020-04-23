import numpy as np

def eval_numerical_gradient_inputs(layer, inputs, in_grads, h=1e-5):
    single_input = False
    if not isinstance(inputs, list):
        single_input = True
        inputs = [inputs]
    grads = [None] * len(inputs)

    for i in range(len(inputs)):
        grads[i] = np.zeros_like(inputs[i])
        it = np.nditer(inputs[i], flags=['multi_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index

            oldval = inputs[i][idx]
            inputs[i][idx] = oldval + h
            if single_input:
                pos = layer.forward(inputs[i]).copy()
            else:
                pos = layer.forward(inputs).copy()
            inputs[i][idx] = oldval - h
            if single_input:
                neg = layer.forward(inputs[i]).copy()
            else:
                neg = layer.forward(inputs).copy()
            inputs[i][idx] = oldval

            grads[i][idx] = np.sum(np.nan_to_num(
                pos - neg) * in_grads) / (2 * h)
            it.iternext()

    if single_input:
        return grads[0]
    else:
        return grads


def eval_numerical_gradient_params(layer, inputs, in_grads, h=1e-5):
    params, _ = layer.get_params('-')
    grads = dict()

    for k, v in params.items():
        grad = np.zeros_like(v)
        it = np.nditer(grad, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index
            oldval = v[idx]
            v[idx] = oldval + h
            pos = layer.forward(inputs).copy()
            v[idx] = oldval - h
            neg = layer.forward(inputs).copy()
            v[idx] = oldval
            grad[idx] = np.sum(np.nan_to_num(pos - neg) * in_grads) / (2 * h)
            it.iternext()

        grads[k] = grad

    return grads


def eval_numerical_gradient_loss(loss, inputs, targets, h=1e-5):
    single_input = False
    if not isinstance(inputs, list):
        single_input = True
        inputs = [inputs]
    grads = [None] * len(inputs)

    for i in range(len(inputs)):
        grads[i] = np.zeros_like(inputs[i])
        it = np.nditer(inputs[i], flags=['multi_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index

            oldval = inputs[i][idx]
            inputs[i][idx] = oldval + h
            if single_input:
                pos = loss.forward(inputs[i], targets)[0].copy()
            else:
                pos = loss.forward(inputs, targets)[0].copy()
            inputs[i][idx] = oldval - h
            if single_input:
                neg = loss.forward(inputs[i], targets)[0].copy()
            else:
                neg = loss.forward(inputs, targets)[0].copy()
            inputs[i][idx] = oldval

            grads[i][idx] = np.sum((pos - neg)) / (2 * h)
            it.iternext()

    if single_input:
        return grads[0]
    else:
        return grads


def check_grads(cacul_grads, numer_grads, threshold=1e-7):
    precise = np.linalg.norm(cacul_grads - numer_grads) / max(
        max(np.linalg.norm(cacul_grads), np.linalg.norm(numer_grads)), threshold)
    return precise


def check_grads_layer(layer, inputs, in_grads):
    map_bool = {
        'True': 'correct',
        'False': 'wrong',
    }

    numer_grads = eval_numerical_gradient_inputs(layer, inputs, in_grads)
    cacul_grads = layer.backward(in_grads, inputs)

    if isinstance(cacul_grads, list):
        for i in range(len(cacul_grads)):
            inputs_result = check_grads(cacul_grads[i], numer_grads[i])
            print('Gradient to input %d:' % (i),
                  map_bool[str(inputs_result < 1e-8)])
    else:
        inputs_result = check_grads(cacul_grads, numer_grads)
        print('Gradient to input:', map_bool[str(inputs_result < 1e-8)])

    if layer.trainable:
        numer_grads = eval_numerical_gradient_params(layer, inputs, in_grads)
        _, cacul_grads = layer.get_params('-')
        for k, v in cacul_grads.items():
            results = check_grads(v, numer_grads[k])
            print('Gradient to %s:' % (k), map_bool[str(results < 1e-8)])


def check_grads_loss(layer, inputs, targets):
    map_bool = {
        'True': 'correct',
        'False': 'wrong',
    }
    numer_grads = eval_numerical_gradient_loss(layer, inputs, targets)
    cacul_grads = layer.backward(inputs, targets)

    # print('<1e-8 will be fine')
    if isinstance(cacul_grads, list):
        for i in range(len(cacul_grads)):
            inputs_result = check_grads(cacul_grads[i], numer_grads[i])
            print('Gradient to input %d:' % (i),
                  map_bool[str(inputs_result < 1e-8)])
    else:
        inputs_result = check_grads(cacul_grads, numer_grads)
        print('Gradient to input:', map_bool[str(inputs_result < 1e-8)])
