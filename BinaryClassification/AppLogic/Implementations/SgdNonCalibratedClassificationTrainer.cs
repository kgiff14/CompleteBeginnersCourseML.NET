using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class SgdNonCalibratedClassificationTrainer : TrainerAbstract<LinearBinaryModelParameters>
{
    public SgdNonCalibratedClassificationTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.SgdNonCalibrated();
}
