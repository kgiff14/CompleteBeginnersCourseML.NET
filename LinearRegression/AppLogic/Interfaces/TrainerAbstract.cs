using LinearRegression.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace LinearRegression.AppLogic.Interfaces;

public abstract class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "regression.mdl");

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

        var dataSet = MlContext.Data.LoadFromTextFile<RealEstateInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.3);
    }

    private IEstimator<ITransformer> CreateDataPipeline()
    {
        var pipeline = MlContext.Transforms.Concatenate("Features", new[] {
                                                        nameof(RealEstateInput.NearestMRT),
                                                        nameof(RealEstateInput.Latitude),
                                                        nameof(RealEstateInput.TransactionDate),
                                                        nameof(RealEstateInput.Longitude),
                                                        nameof(RealEstateInput.NumberOfStores),
                                                        nameof(RealEstateInput.HouseAge)})
                        .Append(MlContext.Transforms.CopyColumns("Label", nameof(RealEstateInput.HousePrice)))
                        .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"));

        return pipeline;
    }
}
