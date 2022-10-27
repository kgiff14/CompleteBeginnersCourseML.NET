using LinearRegression.AppLogic.Interfaces;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace LinearRegression.AppLogic.Implementations;

public class OnlineGradientDescentRegressionTrainer : TrainerAbstract<LinearRegressionModelParameters>
{
    public OnlineGradientDescentRegressionTrainer() => 
        Model = MlContext.Regression.Trainers.OnlineGradientDescent(labelColumnName: "Label", featureColumnName: "Features");
}
