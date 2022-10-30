using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using Sentiment.Models;

namespace Sentiment.AppLogic.Abstracts;

public abstract class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "Sentiment.mdl");

    protected MLContext MlContext;
    protected DataOperationsCatalog.TrainTestData DataSplit;
    protected ITrainerEstimator<BinaryPredictionTransformer<T>, T> Model;
    private ITransformer _trainedModel;

    public TrainerAbstract()
    {
        MlContext = new(0);
    }

    public BinaryClassificationMetrics Evaluate()
    {
        var testSet = _trainedModel.Transform(DataSplit.TestSet);

        return MlContext.BinaryClassification.EvaluateNonCalibrated(testSet);
    }

    public void Fit(string fileName)
    {
        CreateDataSplit(fileName);
        var pipeline = CreateDataPipeline();
        var modelPipeline = pipeline.Append(Model);

        _trainedModel = modelPipeline.Fit(DataSplit.TrainSet);
    }

    public void Save() =>
        MlContext.Model.Save(_trainedModel, DataSplit.TrainSet.Schema, ModelPath);

    private void CreateDataSplit(string fileName)
    {
        if (!File.Exists(fileName))
            throw new FileNotFoundException($"{fileName} was not found");

        var dataSet = MlContext.Data.LoadFromTextFile<ReviewInput>(fileName);
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.3);
    }

    private IEstimator<ITransformer> CreateDataPipeline()
    {
        var pipeline = MlContext.Transforms.Text.FeaturizeText("Features", nameof(ReviewInput.Text))
                    .Append(MlContext.Transforms.CopyColumns("Label", nameof(ReviewInput.Label)))
                    .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                    .AppendCacheCheckpoint(MlContext);

        return pipeline;
    }
}

