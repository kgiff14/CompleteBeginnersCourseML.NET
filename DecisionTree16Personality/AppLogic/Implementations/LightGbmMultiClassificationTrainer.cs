using DecisionTree16Personality.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace DecisionTree16Personality.AppLogic.Implementations;

public class LightGbmMultiClassificationTrainer : TrainerAbstract<OneVersusAllModelParameters>
{
    public LightGbmMultiClassificationTrainer() =>
        Model = MlContext.MulticlassClassification.Trainers.LightGbm();
}
