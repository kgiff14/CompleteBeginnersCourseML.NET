using DecisionTreeRegression.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace DecisionTreeRegression.AppLogic.Implementations;

public class FastForestTrainer : TrainerAbstract<FastForestRegressionModelParameters>
{
    public FastForestTrainer(int numberOfTrees, int numberOfLeaves) =>
        Model = MlContext.Regression.Trainers.FastForest(numberOfTrees: numberOfTrees, numberOfLeaves: numberOfLeaves);
}
