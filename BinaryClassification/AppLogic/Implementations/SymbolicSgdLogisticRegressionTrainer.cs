using BinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace BinaryClassification.AppLogic.Implementations;

public class SymbolicSgdLogisticRegressionTrainer : TrainerAbstract<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
{
    public SymbolicSgdLogisticRegressionTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression();
}
