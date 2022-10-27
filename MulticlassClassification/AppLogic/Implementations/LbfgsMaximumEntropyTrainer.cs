using Microsoft.ML;
using Microsoft.ML.Trainers;
using MulticlassClassification.AppLogic.Abstracts;

namespace MulticlassClassification.AppLogic.Implementations;

public class LbfgsMaximumEntropyTrainer : TrainerAbstract<MaximumEntropyModelParameters>
{
    public LbfgsMaximumEntropyTrainer() =>
        Model = MlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy();
}
