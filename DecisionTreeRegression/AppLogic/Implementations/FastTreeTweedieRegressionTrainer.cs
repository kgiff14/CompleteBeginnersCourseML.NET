using DecisionTreeRegression.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using System.Xml.Linq;

namespace DecisionTreeRegression.AppLogic.Implementations;

public class FastTreeTweedieRegressionTrainer : TrainerAbstract<FastTreeTweedieModelParameters>
{
    public FastTreeTweedieRegressionTrainer(int numberOfLeaves, int numberOfTrees, double learningRate = 0.2) =>
        Model = MlContext.Regression.Trainers.FastTreeTweedie(numberOfLeaves: numberOfLeaves, numberOfTrees: numberOfTrees, learningRate: learningRate);
}
