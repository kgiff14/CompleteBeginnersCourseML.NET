using Microsoft.ML;
using Microsoft.ML.Trainers;
using MulticlassClassification.AppLogic.Abstracts;

namespace MulticlassClassification.AppLogic.Implementations;

public class SdcaMaximumEntropyTrainer : TrainerAbstract<MaximumEntropyModelParameters>
{
    public SdcaMaximumEntropyTrainer() =>
        Model = MlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy();
}
