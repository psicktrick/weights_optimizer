# Weighted Average Optimizer
### Optimizing weighted averages using Genetic Algorithm (DEAP Framework)
This is a generic module for optimizing weighted averages to improve upon results from simple averages. The module uses genetic algorithm to optimize the weights. The evaluation function for the algorithm is defined by the user based on their usecase for eg. this module can be used to assign different weights to different models within an ensemble of models for a regression problem and the objective function can be the root mean squared error of the results as shown in the example in test.py.

#### Usage
Based on the usecase the user needs to define a class which consists of the relevant objective function as a method. The user has to pass an object of this class algong with the number of elements for calculating the weighted average to the WeightsOptimizer class to create an object and thus return the optimized weights.
```python
from ga_weights_optimizer import WeightsOptimizer
n=4 #number of elements in the average
model = CreateModels()
wo = WeightsOptimizer(n, model)
optimized_weights = wo.ga()
```
The configuration for the Genetic Algorithm can be done in config.py
```python
ga={
	"number_of_generations":100,
	"population_size":500,
	"step_size":0.05,
	"weights_lower_lim": 0,
	"weights_upper_limit": 1,
	"checkpoint": False,
	"frequency_checkpoints": 2
}
```

Refer test.py for sample usage for optimizing model weights in ensemble model for regression.
