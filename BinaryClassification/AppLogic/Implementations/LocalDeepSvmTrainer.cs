using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class LocalDeepSvmTrainer : TrainerAbstract<LdSvmModelParameters>
{
    public LocalDeepSvmTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.LdSvm();
}
