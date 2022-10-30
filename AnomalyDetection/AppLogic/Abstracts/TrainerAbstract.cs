using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using AnomalyDetection.Models;

namespace AnomalyDetection.AppLogic.Abstracts;

public abstract class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "Anomaly.mdl");

    protected MLContext MlContext;
    protected DataOperationsCatalog.TrainTestData DataSplit;
    protected ITrainerEstimator<AnomalyPredictionTransformer<T>, T> Model;
    private ITransformer _trainedModel;

    public TrainerAbstract()
    {
        MlContext = new(0);
    }

    public AnomalyDetectionMetrics Evaluate()
    {
        var testSet = _trainedModel.Transform(DataSplit.TestSet);

        return MlContext.AnomalyDetection.Evaluate(testSet);
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

        var dataSet = MlContext.Data.LoadFromTextFile<SkabInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.2);
    }

    private IEstimator<ITransformer> CreateFeatureEngineerPipeline()
    {
        var pipeline = MlContext.Transforms.Concatenate("Features", nameof(SkabInput.Tempurature))
                    .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                    .Append(MlContext.Transforms.CopyColumns("Label", nameof(SkabInput.Anomaly)))
                    .AppendCacheCheckpoint(MlContext);

        return pipeline;
    }
}
