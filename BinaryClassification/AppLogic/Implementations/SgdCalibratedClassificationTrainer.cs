using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class SgdCalibratedClassificationTrainer : TrainerAbstract<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
{
    public SgdCalibratedClassificationTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.SgdCalibrated();
}
