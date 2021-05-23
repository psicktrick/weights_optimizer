from ga_weights_optimizer import WeightsOptimizer



def objective_function(weights):
    return(sum(weights)),


if __name__ == "__main__":
    n=10
    wo = WeightsOptimizer(n, objective_function)
    optimized_weights = wo.ga()
    print(optimized_weights)