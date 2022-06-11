# features is a Matrix, labels and theta are Vectors
# we create the features in a matrix to better calculate the optimization of it
# the result from the equation is the label or labels from the matrix calculation
# the instances are the rows of the matrix
# the vectors rovided are the weights of the features
def cost_function(features, labels, theta)
  # m is counting the number of rows for the features
  m = features.row_count
  # we use the predict function to predict features using the
  # vectors provided which is theta
  # vectors are magnitude * direction where scalar is just magnitude
  predictions = predict(features, theta)
  # squared errors are the errors we will get when predicting the fearures and we would calculate it like so below
  squared_errors = (predictions - labels).map { |error| error**2 }
  # here is the cost function created where we sum up all the squared errors
  # the 1/2m is the number of instances
  # now the cost function is to measure how well our weights are
  # predicting the labels.
  (1.0 / (2.0 * m)) * squared_errors.reduce { |a, b| a + b }
end
