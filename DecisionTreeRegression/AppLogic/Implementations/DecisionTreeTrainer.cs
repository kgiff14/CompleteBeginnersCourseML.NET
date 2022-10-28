using DecisionTreeRegression.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace DecisionTreeRegression.AppLogic.Implementations;

public class DecisionTreeTrainer : TrainerAbstract<FastTreeRegressionModelParameters>
{
    public DecisionTreeTrainer(int numberOfTrees, int numberOfLeaves, double learningRate = 0.2) =>
        Model = MlContext.Regression.Trainers.FastTree(numberOfTrees: numberOfTrees, numberOfLeaves: numberOfLeaves, learningRate: learningRate);
}
