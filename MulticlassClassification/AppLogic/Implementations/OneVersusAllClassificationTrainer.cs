using Microsoft.ML;
using Microsoft.ML.Trainers;
using MulticlassClassification.AppLogic.Abstracts;

namespace MulticlassClassification.AppLogic.Implementations;

public class OneVersusAllClassificationTrainer : TrainerAbstract<OneVersusAllModelParameters>
{
    public OneVersusAllClassificationTrainer() =>
        Model = MlContext.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator: MlContext.BinaryClassification.Trainers.SgdCalibrated());
}
