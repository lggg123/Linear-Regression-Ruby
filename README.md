### Here we are using linear regression to find the terminator battle optimization against humans. 

### A machine learning algorithm would use these instances to come up with the best values for the weights. The best values means: The labels (y) predicted with these weights should be as close as possible to the recorded label. "As close as possible" is where the cost function comes in.

### The Cost Function is measuring the performance of our weights. In linear regression we use the mean squared error as cost function.

### The equation is theta = 1/(number of instances = 2m) SUM (h(x) - y) ^2

### Summing up all the squared errors. In ruby that would be a call to reduce in the example cost_function.rb

# Normal Equation

### Now we add another part to the machine learning algorithm, which is the normal equation. This will reduce our cost, coming up with the best weights for our data set. One caveat: The normal equation is going to geto slower and slower the larger your dataset gets- at that point, you will switch over to gradient descent. Gradient Descent is an iterative approach to figuring out the best weights.

normal equation is as follows

theta = ((x^t)x)^-1)(x^t) * y

((x^t)x)^-1) = the features

(x^t) = transpose
(^-1) = inverse

check normal_equation.rb

The normal equation will give us the following weights:[5.0. 0.675, 17.5]. Use these values to predict the labels.

To summarize:

You "train your model", which means, you minimize the cost function by finding the best weights, with gradient descent or the normal equation. YOu predict vvalues with your hypothesis function. You can measure your predictions with your cost function. 

check linear_regression.rb for the whole code.

Look at the tests

first we setup the equation with @features, @labels, and @initial_theta.Then we test predict with initial theta, test cost with initial theta, test the normal equation, test cost function with optimal theta, then test predict with optimal theta.
