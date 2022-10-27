using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using MulticlassClassification.Model;
using System.Reflection;

namespace MulticlassClassification.AppLogic.Abstracts;

public abstract class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "multiclassClassification.mdl");

    protected MLContext MlContext;
    protected DataOperationsCatalog.TrainTestData DataSplit;
    protected ITrainerEstimator<MulticlassPredictionTransformer<T>, T> Model;
    private ITransformer _trainedModel;

    public TrainerAbstract()
    {
        MlContext = new(0);
    }

    public MulticlassClassificationMetrics Evaluate()
    {
        var testSet = _trainedModel.Transform(DataSplit.TestSet);

        return MlContext.MulticlassClassification.Evaluate(testSet);
    }

    public void Fit(string fileName)
    {
        CreateDataSplit(fileName);
        var pipeline = CreateFeatureEngineerPipeline();
        var modelPipeline = pipeline.Append(Model)
                                    .Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")); ;

        _trainedModel = modelPipeline.Fit(DataSplit.TrainSet);
    }

    public void Save() =>
        MlContext.Model.Save(_trainedModel, DataSplit.TrainSet.Schema, ModelPath);

    private void CreateDataSplit(string fileName)
    {
        if (!File.Exists(fileName))
            throw new FileNotFoundException($"{fileName} was not found");

        var dataSet = MlContext.Data.LoadFromTextFile<DiamondInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.3);
    }

    private IEstimator<ITransformer> CreateFeatureEngineerPipeline()
    {
        var pipeline = MlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(DiamondInput.Label), outputColumnName: "Label")
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(DiamondInput.Clarity), outputColumnName: "ClarityFeature"))
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(DiamondInput.Color), outputColumnName: "ColorFeature"))
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(DiamondInput.Polish), outputColumnName: "PolishFeature"))
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(DiamondInput.Report), outputColumnName: "ReportFeature"))
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(DiamondInput.Symmetry), outputColumnName: "ClarityFeature"))
                .Append(MlContext.Transforms.Concatenate("Features", new[]
                                                            {
                                                                "ClarityFeature",
                                                                "ColorFeature",
                                                                "PolishFeature",
                                                                "ReportFeature",
                                                                "ClarityFeature",
                                                                nameof(DiamondInput.Price)
                                                            }))
                .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"));

        return pipeline;
    }
}
