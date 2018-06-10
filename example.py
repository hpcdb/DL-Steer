from dl-steer import dt_handler, custom_model

#1. Load input data and list L of combinations of hyperparameters

data = dt_handler.read_dataset_info('MNIST')


engine_interface.add_blocks_to_queue('arquivo.json')

model = Sequential()

# Add the input layer
model.add(Dense(number_of_features, kernel_initializer=init_mode, input_dim=number_of_features, activation=activation))
model.add(Dense(n_neurons_hidden_layers, kernel_initializer=init_mode, activation=activation))

...

model = KerasClassifier(build_fn=custom_model, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=n_jobs, scoring="accuracy")

for x in queue:
	grid_result = grid.fit(X, y)

#2.1 fit the model --> este é o que eu tinha chamado de "actual training", mas é ruim chamar assim. Model fitting é mais claro. Essa é a parte computacional mais custosa e é onde acontece o parelismo via tensorflow etc
#2.2 generate model performance (accuracy por exemplo)
#2.3 if accuracy > threshold OR if user requests, stop the loop