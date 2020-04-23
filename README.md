## Python version
- python3

## Python package
- wget
- tensorflow
- keras
- matplotlib
- numpy
- jupyter
- pandas
- nltk

## Structure of the repository

The structure of this repository is shown as below:

```bash
codes/
    data/
        datasets.py     # load dataset, like MNIST
        mnist.npz       # mnist dataset 
        corpus.csv      # for nltk dataset 
        dictionary.csv  # for nltk dataset
    models/             # example models of your tiny deep learning framework
        MNISTNet.py     # example model on MNIST dataset
        SentimentNet.py # example model on nltk dataset
    nn/                 # components of neural networks
        operators.py    # operators
        optimizers.py   # optimizing methods
        layers.py       # layer abstract for CNN and RNN
        loss.py         # loss function for optimization
        model.py        # model abstraction for defining and training models
        initializers.py # initializing methods to initialize parameters (like weights, bias)
        funtional.py    # some helpful function during implementation of training
    utils/              # some additional tools for CNN
        check_grads_cnn.py  # for CNN, help you check your forward function and backward function
        check_grads_rnn.py  # for RNN, help you check your forward function and backward function
        tools.py        # other useful functions for testing the codes
    main.ipynb          # this notebook which calls the functions in other modules/files
    README.MD           # list of dependent libraries
```
