using LinearRegression.AppLogic.Interfaces;
using LinearRegression.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace LinearRegression.AppLogic;

internal class Predictor
{
    private readonly MLContext _mlContext;
    private ITransformer _model;

    internal Predictor()
    {
        _mlContext = new(0);
    }

    internal void Predict<T>(RealEstateInput input, TrainerAbstract<T> trainer) where T : class
    {
        Console.WriteLine($"\n\n-------------------------------------------------------------------\n\t{trainer}\n-------------------------------------------------------------------\n");

        string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Assets\", "Real estate.csv");
        trainer.Fit(path.ToString());

        var modelMetrics = trainer.Evaluate();
        RecordMetrics(modelMetrics);

        trainer.Save();
        LoadModel(trainer);

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<RealEstateInput, RealEstateOutput>(_model);
        var prediction = predictionEngine.Predict(input);
        Console.WriteLine($"\nPrediction: {prediction.Price:#.##}");
    }

    private void RecordMetrics(RegressionMetrics modelMetrics)
    {
        Console.WriteLine($"\nMean Absolute Error: {modelMetrics.MeanSquaredError: #.##}");
        Console.WriteLine($"\nMean Squared Error: {modelMetrics.MeanSquaredError: #.##}");
        Console.WriteLine($"\nRoot Mean Squared Error: {modelMetrics.RootMeanSquaredError: #.##}");
        Console.WriteLine($"\nLoss Function: {modelMetrics.LossFunction: #.##}");
        Console.WriteLine($"\nR Squared: {modelMetrics.RSquared: #.##}");
    }

    private void LoadModel<T>(TrainerAbstract<T> trainer) where T : class
    {
        if (!File.Exists(trainer.ModelPath))
            throw new FileNotFoundException($"{trainer.ModelPath} was not found.");

        using var stream = new FileStream(trainer.ModelPath, FileMode.Open, FileAccess.Read);

        _model = _mlContext.Model.Load(stream, out var _);

        if (_model is null)
            throw new Exception("Failed to load model");
    }
}
