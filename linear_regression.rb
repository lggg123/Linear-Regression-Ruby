require 'minitest/autorun'
require 'matrix'

# If we wanted to call this module we simply call the methods within the module and display it onto the console.
# To summarize everything we create the prediction to create the cost function which will give us the sum of all of our Js as we see with the 
# reduce method at the end.
# the sum of Js is the sum of all of our weights we find the errors
# and discrepenacies by subtracting predictions - labels and then squareit. Then we try to find the sum of all of our Js.
# We train the model by creating normal equation that is going to give
# the optimal weights and then can use those weights to get the sum of our Js again and get the optimal outcome that would be to occur.
# the goal of linear regression is to ge the cost as close to 0 as possible and the only way is to find the optimal weights for the data. The cost is the cost for our initial weights by the way.
module LinearRegression
  def predict(features, theta)
    features * theta
  end
  
  def cost_function(features, labels, theta)
    m = features.row_count
    predictions = predict(features, theta)
    squared_errors = (predictions - labels).map { |error| error**2 }
    (1.0/(2.0*m)) * squared_errors.reduce { |a, b| a + b }
  end
  
  def normal_equation(features, labels)
    (features.transpose * features).inverse * features.transpose * labels
  end
end

class TestLinearRegression < MiniTest::Test
  include LinearRegression

  def setup
    @features      = Matrix[[10, 1000, 10], [5, 800, 2], [12, 2500, 3]]
    @labels        = Vector[900, 600, 1800]
    # 900 represents the losses for 10 terminators, 1000 humans, and 10 dogs# 600 reps  the human losses for 5 terminators, 800 humans, andn 2 dogs
    # 1800 represents the human losses fo 12 terminators, 2500 humans, 3 dogs
    # initial_theta is the initial weights to the equation given.
    @initial_theta = Vector[200, -0.1, -10]
  end 

  def test_predict_with_initial_theta
    assert_equal Vector[1800, 900, 2120], predict(@features, @initial_theta)
  end
  
  def test_cost_with_initial_theta
    assert_equal 167_066.67, cost_function(@features, @labels, @initial_theta).round(2)
  end

  def test_normal_equation
    assert_equal Vector[5.0, 0.675, 17.5], normal_equation(@features, @labels)
  end

  def test_cost_function_with_optimal_theta
    optimal_theta = Vector[5.0, 0.675, 17.5]
    assert_equal 0, cost_function(@features, @labels, optimal_theta)
  end

  def test_predict_with_optimal_theta
    optimal_theta = normal_equation(@features, @labels)
    assert_equal Vector[900, 600, 1800], predict(@features, optimal_theta)
  end
end
