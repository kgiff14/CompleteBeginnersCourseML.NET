using Microsoft.ML.Data;
using Microsoft.ML;
using Forecasting.Models;
using Forecasting.AppLogic.Abstracts;
using Microsoft.ML.Transforms.TimeSeries;

namespace Forecasting.AppLogic;

internal class Predictor
{
    private readonly MLContext _mlContext;
    private ITransformer _model;
    public static string ModelPath => Path.Combine(AppContext.BaseDirectory, "forecast.zip");

    internal Predictor()
    {
        _mlContext = new(0);
    }

    internal void Predict(TrainerAbstract trainer)
    {
        Console.WriteLine($"\n\n-------------------------------------------------------------------\n\t{trainer}\n-------------------------------------------------------------------\n");

        string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Assets\", "BTC-USD.csv");
        trainer.Fit(path.ToString());
        var predictionEngine = TrainerAbstract._trainedModel.CreateTimeSeriesEngine<BitcoinInput, BitcoinOutput>(_mlContext);

        Save(predictionEngine);
        LoadModel(trainer);

        var predictionEngineCopy = _model.CreateTimeSeriesEngine<BitcoinInput, BitcoinOutput>(_mlContext);
        var prediction = predictionEngineCopy.Predict();

        for (var i = 0; i < prediction.Open.Length; i++)
        {
            Console.WriteLine($"\nDay: {i + 1}");
            Console.WriteLine($"\nPrediction: {prediction.Open[i]:#.##}");
            Console.WriteLine($"\nPrediction Upper: {prediction.Open_UB[i]:#.##}");
            Console.WriteLine($"\nPrediction Lower: {prediction.Open_LB[i]:#.##}");

        }
    }

    private void LoadModel(TrainerAbstract trainer)
    {
        if (!File.Exists(trainer.ModelPath))
            throw new FileNotFoundException($"{trainer.ModelPath} was not found.");

        using var stream = new FileStream(trainer.ModelPath, FileMode.Open, FileAccess.Read);

        _model = _mlContext.Model.Load(stream, out var _);

        if (_model is null)
            throw new Exception("Failed to load model");
    }

    private void Save(TimeSeriesPredictionEngine<BitcoinInput, BitcoinOutput> predictionEngine)
    {
        predictionEngine.CheckPoint(_mlContext, ModelPath);
    }
}
