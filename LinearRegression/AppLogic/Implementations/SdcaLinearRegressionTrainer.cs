using LinearRegression.AppLogic.Interfaces;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace LinearRegression.AppLogic.Implementations;

internal class SdcaLinearRegressionTrainer : TrainerAbstract<LinearRegressionModelParameters>
{
    internal SdcaLinearRegressionTrainer() : base() => 
        Model = MlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
}
