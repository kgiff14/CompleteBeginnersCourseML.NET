using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using Recommendation.Models;
using Microsoft.ML.Trainers.Recommender;

namespace Recommendation.AppLogic.Abstracts;

public abstract class TrainerAbstract
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "Recommendation.mdl");

    protected MLContext MlContext;
    protected DataOperationsCatalog.TrainTestData DataSplit;
    protected ITrainerEstimator<MatrixFactorizationPredictionTransformer, MatrixFactorizationModelParameters> Model;
    private ITransformer _trainedModel;

    public TrainerAbstract()
    {
        MlContext = new(0);
    }

    public RegressionMetrics Evaluate()
    {
        var testSet = _trainedModel.Transform(DataSplit.TestSet);

        return MlContext.Regression.Evaluate(testSet);
    }

    public void Fit(string fileName)
    {
        CreateDataSplit(fileName);
        var pipeline = CreateFeatureEngineerPipeline();
        var modelPipeline = pipeline.Append(Model);

        _trainedModel = modelPipeline.Fit(DataSplit.TrainSet);
    }

    public void Save() =>
        MlContext.Model.Save(_trainedModel, DataSplit.TrainSet.Schema, ModelPath);

    private void CreateDataSplit(string fileName)
    {
        if (!File.Exists(fileName))
            throw new FileNotFoundException($"{fileName} was not found");

        var dataSet = MlContext.Data.LoadFromTextFile<AnimeInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.2);
    }

    private IEstimator<ITransformer> CreateFeatureEngineerPipeline()
    {
        var pipeline = MlContext.Transforms.Conversion.MapValueToKey("UserFeature", nameof(AnimeInput.UserId))
                    .Append(MlContext.Transforms.Conversion.MapValueToKey("AnimeFeature", nameof(AnimeInput.AnimeId)))
                    .Append(MlContext.Transforms.CopyColumns("Label", nameof(AnimeInput.Rating)))
                    .AppendCacheCheckpoint(MlContext);

        return pipeline;
    }
}