using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class SdcaNonCalibratedTrainer : TrainerAbstract<LinearBinaryModelParameters>
{
    public SdcaNonCalibratedTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.SdcaNonCalibrated();
}
