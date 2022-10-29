using LinearRegression.AppLogic.Interfaces;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace LinearRegression.AppLogic.Implementations;

public class OlsRegressionTrainer : TrainerAbstract<OlsModelParameters>
{
    public OlsRegressionTrainer() =>
        Model = MlContext.Regression.Trainers.Ols();
}
