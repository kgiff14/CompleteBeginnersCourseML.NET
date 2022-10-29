using Microsoft.ML.Data;
using Microsoft.ML;
using Clustering.Models;
using Clustering.AppLogic.Abstracts;

namespace Clustering.AppLogic;

internal class Predictor
{
    private readonly MLContext _mlContext;
    private ITransformer _model;

    internal Predictor()
    {
        _mlContext = new(0);
    }

    internal void Predict<T>(MallInput input, TrainerAbstract<T> trainer) where T : class
    {
        Console.WriteLine($"\n\n*************************************************\n*\t{trainer}\n*\n*");

        string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Assets\", "Mall_Customers.csv");
        trainer.Fit(path.ToString());

        var modelMetrics = trainer.Evaluate();
        RecordMetrics(modelMetrics);

        trainer.Save();
        LoadModel(trainer);

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<MallInput, MallOutput>(_model);
        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"*");
        Console.WriteLine($"*-------------------------------");
        Console.WriteLine($"*");
        Console.WriteLine($"*\tLabel: {prediction.Label}");
        Console.WriteLine($"*");
        Console.WriteLine($"*************************************************");
    }

    private void RecordMetrics(ClusteringMetrics modelMetrics)
    {
        Console.WriteLine($"*\tAverage Distance: {modelMetrics.AverageDistance: #.##}");
        Console.WriteLine($"*\tDavies Bouldin Index: {modelMetrics.DaviesBouldinIndex}");
        Console.WriteLine($"*\tNormalized Mutual information: {modelMetrics.NormalizedMutualInformation: #.##}");
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
