using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class AveragedPerceptronClassificationTrainer : TrainerAbstract<LinearBinaryModelParameters>
{
    public AveragedPerceptronClassificationTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.AveragedPerceptron();
}
