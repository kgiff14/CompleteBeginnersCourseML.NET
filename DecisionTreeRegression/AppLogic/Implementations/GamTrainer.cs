using DecisionTreeRegression.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace DecisionTreeRegression.AppLogic.Implementations;

public class GamTrainer : TrainerAbstract<GamRegressionModelParameters>
{
    public GamTrainer() =>
        Model = MlContext.Regression.Trainers.Gam();
}
