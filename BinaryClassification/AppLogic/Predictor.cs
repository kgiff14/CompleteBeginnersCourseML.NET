using BinaryClassification.AppLogic.Abstracts;
using BinaryClassification.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BinaryClassification.AppLogic;

internal class Predictor
{
    private readonly MLContext _mlContext;
    private ITransformer _model;

    internal Predictor()
    {
        _mlContext = new(0);
    }

    internal void Predict<T>(BreastCancerInput input, TrainerAbstract<T> trainer) where T : class
    {
        Console.WriteLine($"\n\n*************************************************\n*\t{trainer}\n*\n*");

        string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Assets\", "breast-cancer.csv");
        trainer.Fit(path.ToString());

        var modelMetrics = trainer.Evaluate();
        RecordMetrics(modelMetrics);

        trainer.Save();
        LoadModel(trainer);

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<BreastCancerInput, BreastCancerOutput>(_model);
        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"*");
        Console.WriteLine($"*-------------------------------");
        Console.WriteLine($"*");
        Console.WriteLine($"*\tIs Malignant: {prediction.IsMalignant}");
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