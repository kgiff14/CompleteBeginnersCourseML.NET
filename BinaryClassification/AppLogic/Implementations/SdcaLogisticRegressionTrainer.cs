using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class SdcaLogisticRegressionTrainer : TrainerAbstract<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
{
    public SdcaLogisticRegressionTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
}
