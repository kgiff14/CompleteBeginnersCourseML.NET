using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using BinaryClassification.Models;

namespace BinaryClassification.AppLogic.Abstracts;

public abstract class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "binaryClassification.mdl");

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

        var dataSet = MlContext.Data.LoadFromTextFile<BreastCancerInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.3);
    }

    private IEstimator<ITransformer> CreateDataPipeline()
    {
        var pipeline = MlContext.Transforms.Concatenate("Features", new[]{
                                                nameof(BreastCancerInput.RadiusMean),
                                                nameof(BreastCancerInput.TextureMean),
                                                nameof(BreastCancerInput.PerimeterMean),
                                                nameof(BreastCancerInput.AreaMean),
                                                nameof(BreastCancerInput.SmoothnessMean),
                                                nameof(BreastCancerInput.CompactnessMean),
                                                nameof(BreastCancerInput.ConcavityMean),
                                                nameof(BreastCancerInput.ConcaveMean),
                                                nameof(BreastCancerInput.SymmetryMean),
                                                nameof(BreastCancerInput.FractialDimensionMean),
                                                nameof(BreastCancerInput.RadiusSe),
                                                nameof(BreastCancerInput.TextureSe),
                                                nameof(BreastCancerInput.PerimeterSe),
                                                nameof(BreastCancerInput.AreaSe),
                                                nameof(BreastCancerInput.SmoothnessSe),
                                                nameof(BreastCancerInput.CompactnessSe),
                                                nameof(BreastCancerInput.ConcaveSe),
                                                nameof(BreastCancerInput.ConcavitySe),
                                                nameof(BreastCancerInput.SymmetrySe),
                                                nameof(BreastCancerInput.FractialDimensionSe),
                                                nameof(BreastCancerInput.RadiusWorst),
                                                nameof(BreastCancerInput.TextureWorst),
                                                nameof(BreastCancerInput.PerimeterWorst),
                                                nameof(BreastCancerInput.AreaWorst),
                                                nameof(BreastCancerInput.SmoothnessWorst),
                                                nameof(BreastCancerInput.CompactnessWorst),
                                                nameof(BreastCancerInput.ConcaveWorst),
                                                nameof(BreastCancerInput.ConcavityWorst),
                                                nameof(BreastCancerInput.SymmetryWorst),
                                                nameof(BreastCancerInput.FractialDimensionWorst)
                                                })
                    .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                    .AppendCacheCheckpoint(MlContext);

        return pipeline;
    }
}
