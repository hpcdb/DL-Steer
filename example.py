from dl-steer import dt_handler , coordinator , custom_model
from keras.wrappers.scikit_learn import KerasClassifier
from spark_sklearn import GridSearchCV
from keras.models import Sequential

#1. Load input data
data = dt_handler.read_dataset('input_data.csv')


model = Sequential()

# Add the input layer
#model.add(Dense(number_of_features, kernel_initializer=init_mode, input_dim=number_of_features, activation=activation))
#model.add(Dense(n_neurons_hidden_layers, kernel_initializer=init_mode, activation=activation))

...

model = KerasClassifier(build_fn=custom_model, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=n_jobs, scoring="accuracy")

queue = coordinator.get_queue()

X, y = data['X'], data['y']

for x in queue:
	grid_result = grid.fit(X, y)