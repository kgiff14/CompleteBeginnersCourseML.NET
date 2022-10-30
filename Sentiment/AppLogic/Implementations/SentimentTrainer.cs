using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;
using Sentiment.AppLogic.Abstracts;

namespace Sentiment.AppLogic.Implementations;

public class SentimentTrainer : TrainerAbstract<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
{
    public SentimentTrainer() =>
        Model = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
}
