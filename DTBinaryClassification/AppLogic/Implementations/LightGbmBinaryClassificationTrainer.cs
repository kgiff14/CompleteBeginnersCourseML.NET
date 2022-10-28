using DTBinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.LightGbm;

namespace DTBinaryClassification.AppLogic.Implementations;

public class LightGbmBinaryClassificationTrainer : TrainerAbstract<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>
{
    public LightGbmBinaryClassificationTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.LightGbm();
}
