using Microsoft.ML;
using Microsoft.ML.Trainers;
using MulticlassClassification.AppLogic.Abstracts;

namespace MulticlassClassification.AppLogic.Implementations;

public class PairwiseTrainer : TrainerAbstract<PairwiseCouplingModelParameters>
{
    public PairwiseTrainer() =>
        Model = MlContext.MulticlassClassification.Trainers.PairwiseCoupling(MlContext.BinaryClassification.Trainers.SdcaLogisticRegression());
}
