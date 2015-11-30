from sklearn.ensemble import RandomForestClassifier as Classifier
from sklearn.metrics import zero_one_loss as loss
import pandas as pd
import numpy as np

# Seed numpy rng
np.random.seed(123)

train_x = pd.read_csv('train_x.csv')
train_y = pd.read_csv('train_y.csv')

split = int(0.8 * len(train_x))
train_x, valid_x = train_x[:split], train_x[split:]
train_y, valid_y = train_y[:split], train_y[split:]

# Convert train_y to a 1d array to silence sklearn conversion warnings
train_y = train_y.as_matrix().reshape((train_y.shape[0],))

test_x = pd.read_csv('test_x.csv')
test_y = pd.read_csv('test_y.csv')

# Using CV, discovered n=526 to be best (rng is seeded, results will not change)
n_estimators = [ 526 ]
best_error = float('Inf')
best_model = None
best_n = None

for n in n_estimators:
	clf = Classifier(n_estimators=n, criterion='entropy', n_jobs=-1, random_state=123)
	clf.fit(train_x, train_y)

	predictions = clf.predict(valid_x)
	error = loss(valid_y, predictions)
	print "Validation error: " + str(error)

	if error < best_error:
		best_error = error
		best_model = clf
		best_n = n

print "Best n: " + str(best_n)

predictions = best_model.predict(test_x)
test_error = loss(test_y, predictions)
print "Test error: " + str(test_error)

feature_names = train_x.columns.values.tolist()

features_errors = []
for feature in feature_names:
	errors = []
	# Get an average over 5 shuffles to smooth out results
	for i in range(5):
		test_x_modified = test_x.copy()
		np.random.shuffle(test_x_modified[feature])
		predictions = best_model.predict(test_x_modified)
		error = loss(test_y, predictions)
		errors.append(error)

	avg_shuffled_error = sum(errors) / 5.
	features_errors.append((feature, avg_shuffled_error - test_error))

features_errors = sorted(features_errors, key=lambda x: x[1], reverse=True)

print "Feature's error differences (error_shuffled - test_error):"
for idx, (feature, error_diff) in enumerate(features_errors):
	print str(idx + 1) + ") " + feature + ": " + str(error_diff)
