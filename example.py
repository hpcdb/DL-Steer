from dl_steer import dt_handler , coordinator , custom_model, engine_interface, provenance
from keras.wrappers.scikit_learn import KerasClassifier
from spark_sklearn import GridSearchCV
from keras.models import Sequential

data = dt_handler.read_dataset('input_data.csv')

...


model = KerasClassifier(build_fn=custom_model.get_model(), verbose=0)
X, y = data['X'], data['y']

queue = coordinator.get_queue()
for hyperparameter_combination in queue:
    provenance.persist(hyperparameter_combination)
    grid = GridSearchCV(estimator=model, param_grid=hyperparameter_combination, n_jobs=-1, scoring="accuracy")
    grid_result = grid.fit(X, y)
    provenance.persist(grid_result)
    #The method below verifies if user steered the queue. If yes, it reloads the queue accordingly.
    queue.checkSteering()
