using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using Clustering.Models;

namespace Clustering.AppLogic.Abstracts;

public abstract class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "Clustering.mdl");

    protected MLContext MlContext;
    protected DataOperationsCatalog.TrainTestData DataSplit;
    protected ITrainerEstimator<ClusteringPredictionTransformer<T>, T> Model;
    private ITransformer _trainedModel;

    public TrainerAbstract()
    {
        MlContext = new(0);
    }

    public ClusteringMetrics Evaluate()
    {
        var testSet = _trainedModel.Transform(DataSplit.TestSet);

        return MlContext.Clustering.Evaluate(data: testSet,
                                            labelColumnName: "PredictedLabel",
                                            scoreColumnName: "Score",
                                            featureColumnName: "Features");
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

        var dataSet = MlContext.Data.LoadFromTextFile<MallInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.2);
    }

    private IEstimator<ITransformer> CreateFeatureEngineerPipeline()
    {
        var pipeline = MlContext.Transforms.Text.FeaturizeText("GenderFeature", nameof(MallInput.Gender))
                    .Append(MlContext.Transforms.Concatenate("Features", new[]
                                                    {
                                                        "GenderFeature",
                                                        nameof(MallInput.Age),
                                                        nameof(MallInput.CustId),
                                                        nameof(MallInput.Income),
                                                        nameof(MallInput.Spending),
                                                    }))
                    .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                    .AppendCacheCheckpoint(MlContext);

        return pipeline;
    }
}
