using DTBinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace DTBinaryClassification.AppLogic.Implementations;

public class GamTrainer : TrainerAbstract<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>
{
    public GamTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.Gam();
}
