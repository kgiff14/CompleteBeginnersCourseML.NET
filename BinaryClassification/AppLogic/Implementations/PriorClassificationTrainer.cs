using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class PriorClassificationTrainer : TrainerAbstract<PriorModelParameters>
{
    public PriorClassificationTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.Prior();
}
