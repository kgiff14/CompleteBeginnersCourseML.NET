using Microsoft.ML.Data;
using Microsoft.ML;
using Sentiment.Models;
using Sentiment.AppLogic.Abstracts;

namespace Sentiment.AppLogic;

internal class Predictor
{
    private readonly MLContext _mlContext;
    private ITransformer _model;

    internal Predictor()
    {
        _mlContext = new(0);
    }

    internal void Predict<T>(ReviewInput input, TrainerAbstract<T> trainer) where T : class
    {
        Console.WriteLine($"\n\n*************************************************\n*\t{trainer}\n*\n*");

        string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Assets\", "movie_reviews.txt");
        trainer.Fit(path.ToString());

        var modelMetrics = trainer.Evaluate();
        RecordMetrics(modelMetrics);

        trainer.Save();
        LoadModel(trainer);

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<ReviewInput, ReviewOutput>(_model);
        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"*");
        Console.WriteLine($"*-------------------------------");
        Console.WriteLine($"*");
        Console.WriteLine($"*\tIs Positive: {prediction.IsPositive}");
        Console.WriteLine($"*");
        Console.WriteLine($"*************************************************");
    }

    private void RecordMetrics(BinaryClassificationMetrics modelMetrics)
    {
        Console.WriteLine($"*\tAccuracy: {modelMetrics.Accuracy: 0.##}");
        Console.WriteLine($"*\tF1 Score: {modelMetrics.F1Score: #.##}");
        Console.WriteLine($"*\tPositive Precision: {modelMetrics.PositivePrecision: #.##}");
        Console.WriteLine($"*\tNegative Precision: {modelMetrics.NegativePrecision: 0.##}");
        Console.WriteLine($"*\tPositive Recall: {modelMetrics.PositiveRecall: #.##}");
        Console.WriteLine($"*\tNegative Recall: {modelMetrics.NegativeRecall: #.##}");
        Console.WriteLine($"*\tArea Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve: #.##}");
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