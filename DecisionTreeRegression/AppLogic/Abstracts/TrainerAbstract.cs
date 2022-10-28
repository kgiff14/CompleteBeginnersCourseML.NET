using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using DecisionTreeRegression.Models;

namespace DecisionTreeRegression.AppLogic.Abstracts;

public abstract class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "decisionTreeRegression.mdl");

    protected MLContext MlContext;
    protected DataOperationsCatalog.TrainTestData DataSplit;
    protected ITrainerEstimator<RegressionPredictionTransformer<T>, T> Model;
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

        var dataSet = MlContext.Data.LoadFromTextFile<EnergyEfficiencyInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.3);
    }

    private IEstimator<ITransformer> CreateFeatureEngineerPipeline()
    {
        var pipeline = MlContext.Transforms.CopyColumns("Label", nameof(EnergyEfficiencyInput.HeatingLoad))
                    .Append(MlContext.Transforms.Concatenate("Features", new[]
                    {
                        nameof(EnergyEfficiencyInput.Compactness),
                        nameof(EnergyEfficiencyInput.SurfaceArea),
                        nameof(EnergyEfficiencyInput.WallArea),
                        nameof(EnergyEfficiencyInput.RoofArea),
                        nameof(EnergyEfficiencyInput.Height),
                        nameof(EnergyEfficiencyInput.Orientation),
                        nameof(EnergyEfficiencyInput.GlazingArea),
                        nameof(EnergyEfficiencyInput.GlazingAreaDistribution),
                    }))
                    .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                    .AppendCacheCheckpoint(MlContext);

        return pipeline;
    }
}
