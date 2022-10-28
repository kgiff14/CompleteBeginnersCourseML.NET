using DecisionTreeRegression.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;

namespace DecisionTreeRegression.AppLogic.Implementations;

public class LightGbmTrainer : TrainerAbstract<LightGbmRegressionModelParameters>
{
    public LightGbmTrainer() =>
        Model = MlContext.Regression.Trainers.LightGbm();
}
