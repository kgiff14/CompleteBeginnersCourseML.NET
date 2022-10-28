using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using DTBinaryClassification.Models;

namespace DTBinaryClassification.AppLogic.Abstracts;

public abstract class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "DTBinaryClassification.mdl");

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

        var dataSet = MlContext.Data.LoadFromTextFile<AutismInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.3);
    }

    private IEstimator<ITransformer> CreateFeatureEngineerPipeline()
    {
        var pipeline = MlContext.Transforms.Text.FeaturizeText(outputColumnName: "GenderFeature", inputColumnName: nameof(AutismInput.Gender))
                    .Append(MlContext.Transforms.Text.FeaturizeText(outputColumnName: "EthinicityFeature", inputColumnName: nameof(AutismInput.Ethnicity)))
                    .Append(MlContext.Transforms.Text.FeaturizeText(outputColumnName: "CountryFeature", inputColumnName: nameof(AutismInput.Country)))
                    .Append(MlContext.Transforms.Text.FeaturizeText(outputColumnName: "JundiceFeature", inputColumnName: nameof(AutismInput.Jundice)))
                    .Append(MlContext.Transforms.Text.FeaturizeText(outputColumnName: "AutismFeature", inputColumnName: nameof(AutismInput.Autism)))
                    .Append(MlContext.Transforms.Text.FeaturizeText(outputColumnName: "UsedAppFeature", inputColumnName: nameof(AutismInput.UsedApp)))
                    .Append(MlContext.Transforms.Concatenate("Features", new[]{
                                                                nameof(AutismInput.A1),
                                                                nameof(AutismInput.A2),
                                                                nameof(AutismInput.A3),
                                                                nameof(AutismInput.A4),
                                                                nameof(AutismInput.A5),
                                                                nameof(AutismInput.A6),
                                                                nameof(AutismInput.A7),
                                                                nameof(AutismInput.A8),
                                                                nameof(AutismInput.A9),
                                                                nameof(AutismInput.A10),
                                                                nameof(AutismInput.Age),
                                                                "GenderFeature",
                                                                "JundiceFeature",
                                                                "AutismFeature",
                                                                "EthinicityFeature",
                                                                "CountryFeature"
                                                                }))
                    .Append(MlContext.Transforms.CopyColumns("Label", nameof(AutismInput.Class)))
                    .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                    .AppendCacheCheckpoint(MlContext);

        return pipeline;
    }
}
