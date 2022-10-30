using Microsoft.ML.Data;
using Microsoft.ML;
using ImageClassification.Models;
using ImageClassification.AppLogic.Abstracts;

namespace ImageClassification.AppLogic;

internal class Predictor
{
    private readonly MLContext _mlContext;
    private ITransformer _model;

    internal Predictor()
    {
        _mlContext = new(0);
    }

    internal void Predict<T>(ImageInput input, TrainerAbstract<T> trainer) where T : class
    {
        Console.WriteLine($"\n\n*************************************************\n*\t{trainer}\n*\n*");

        trainer.Fit();

        var modelMetrics = trainer.Evaluate();
        RecordMetrics(modelMetrics);

        trainer.Save();
        LoadModel(trainer);

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<ImageInput, ImageOutput>(_model);
        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"*");
        Console.WriteLine($"*-------------------------------");
        Console.WriteLine($"*");
        Console.WriteLine($"*\tImage prediction: {prediction.PredictedLabel}");
        Console.WriteLine($"*");
        Console.WriteLine($"*************************************************");
    }

    private void RecordMetrics(MulticlassClassificationMetrics modelMetrics)
    {
        Console.WriteLine($"*\tMacro Accuracy: {modelMetrics.MacroAccuracy: #.##}");
        Console.WriteLine($"*\tMicro Accuracy: {modelMetrics.MicroAccuracy: #.##}");
        Console.WriteLine($"*\tLog Loss: {modelMetrics.LogLoss: #.##}");
        Console.WriteLine($"*\tLog Loss Reduction: {modelMetrics.LogLossReduction: 0.##}");
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