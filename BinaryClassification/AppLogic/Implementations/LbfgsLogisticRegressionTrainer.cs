using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class LbfgsLogisticRegressionTrainer : TrainerAbstract<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
{
    public LbfgsLogisticRegressionTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.LbfgsLogisticRegression();
}
