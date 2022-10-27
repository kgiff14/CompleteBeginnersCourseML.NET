using Microsoft.ML;
using Microsoft.ML.Trainers;
using MulticlassClassification.AppLogic.Abstracts;

namespace MulticlassClassification.AppLogic.Implementations;

public class NaiveBayesTrainer : TrainerAbstract<NaiveBayesMulticlassModelParameters>
{
    public NaiveBayesTrainer() =>
        Model = MlContext.MulticlassClassification.Trainers.NaiveBayes();
}
