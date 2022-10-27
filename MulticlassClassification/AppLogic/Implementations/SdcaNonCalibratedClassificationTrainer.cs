using Microsoft.ML;
using Microsoft.ML.Trainers;
using MulticlassClassification.AppLogic.Abstracts;

namespace MulticlassClassification.AppLogic.Implementations;

public class SdcaNonCalibratedClassificationTrainer : TrainerAbstract<LinearMulticlassModelParameters>
{
    public SdcaNonCalibratedClassificationTrainer() =>
        Model = MlContext.MulticlassClassification.Trainers.SdcaNonCalibrated();
}
