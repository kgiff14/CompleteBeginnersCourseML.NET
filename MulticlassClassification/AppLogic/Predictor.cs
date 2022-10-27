using Microsoft.ML.Data;
using Microsoft.ML;
using MulticlassClassification.Model;
using MulticlassClassification.AppLogic.Abstracts;

namespace MulticlassClassification.AppLogic;

internal class Predictor
{
    private readonly MLContext _mlContext;
    private ITransformer _model;

    internal Predictor()
    {
        _mlContext = new(0);
    }

    internal void Predict<T>(DiamondInput input, TrainerAbstract<T> trainer) where T : class
    {
        Console.WriteLine($"\n\n*************************************************\n*\t{trainer}\n*\n*");

        string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Assets\", "diamond.csv");
        trainer.Fit(path.ToString());

        var modelMetrics = trainer.Evaluate();
        RecordMetrics(modelMetrics);

        trainer.Save();
        LoadModel(trainer);

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<DiamondInput, DiamondOutput>(_model);
        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"*");
        Console.WriteLine($"*-------------------------------");
        Console.WriteLine($"*");
        Console.WriteLine($"*\tCut: {prediction.PredictedLabel}");
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
