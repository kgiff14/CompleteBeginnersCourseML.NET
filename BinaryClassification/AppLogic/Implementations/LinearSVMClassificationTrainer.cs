using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class LinearSVMClassificationTrainer : TrainerAbstract<LinearBinaryModelParameters>
{
    public LinearSVMClassificationTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.LinearSvm();
}
