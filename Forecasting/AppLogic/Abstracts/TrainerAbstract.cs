using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using Forecasting.Models;
using Microsoft.ML.Transforms.TimeSeries;

namespace Forecasting.AppLogic.Abstracts;

public abstract class TrainerAbstract
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "forecast.zip");

    protected MLContext MlContext;
    protected DataOperationsCatalog.TrainTestData DataSplit;
    protected IEstimator<ITransformer> Model;
    internal static ITransformer _trainedModel;

    public TrainerAbstract()
    {
        MlContext = new(0);
    }

    public void Fit(string fileName)
    {
        CreateDataSplit(fileName);

        _trainedModel = Model.Fit(DataSplit.TrainSet);
    }

    private void CreateDataSplit(string fileName)
    {
        if (!File.Exists(fileName))
            throw new FileNotFoundException($"{fileName} was not found");

        var dataSet = MlContext.Data.LoadFromTextFile<BitcoinInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.2);
    }
}