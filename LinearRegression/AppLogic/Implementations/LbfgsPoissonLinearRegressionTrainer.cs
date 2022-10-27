using LinearRegression.AppLogic.Interfaces;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace LinearRegression.AppLogic.Implementations;

public class LbfgsPoissonLinearRegressionTrainer : TrainerAbstract<PoissonRegressionModelParameters>
{
    public LbfgsPoissonLinearRegressionTrainer()
    {
        Model = MlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Label", featureColumnName: "Features");
    }
}
