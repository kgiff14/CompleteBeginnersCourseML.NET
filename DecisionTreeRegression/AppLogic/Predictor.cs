using Microsoft.ML.Data;
using Microsoft.ML;
using DecisionTreeRegression.Models;
using DecisionTreeRegression.AppLogic.Abstracts;

namespace DecisionTreeRegression.AppLogic;

internal class Predictor
{
    private readonly MLContext _mlContext;
    private ITransformer _model;

    internal Predictor()
    {
        _mlContext = new(0);
    }

    internal void Predict<T>(EnergyEfficiencyInput input, TrainerAbstract<T> trainer) where T : class
    {
        Console.WriteLine($"\n\n*************************************************\n*\t{trainer}\n*\n*");

        string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Assets\", "energy_efficiency_data.csv");
        trainer.Fit(path.ToString());

        var modelMetrics = trainer.Evaluate();
        RecordMetrics(modelMetrics);

        trainer.Save();
        LoadModel(trainer);

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<EnergyEfficiencyInput, EnergyEfficiencyOutput>(_model);
        var prediction = predictionEngine.Predict(input);

        Console.WriteLine($"*");
        Console.WriteLine($"*-------------------------------");
        Console.WriteLine($"*");
        Console.WriteLine($"*\tHeating Load: {prediction.HeatingLoad}");
        Console.WriteLine($"*");
        Console.WriteLine($"*************************************************");
    }

    private void RecordMetrics(RegressionMetrics modelMetrics)
    {
        Console.WriteLine($"*\tMean Absolute Error: {modelMetrics.MeanSquaredError: #.##}");
        Console.WriteLine($"*\tMean Squared Error: {modelMetrics.MeanSquaredError: #.##}");
        Console.WriteLine($"*\tRoot Mean Squared Error: {modelMetrics.RootMeanSquaredError: #.##}");
        Console.WriteLine($"*\tLoss Function: {modelMetrics.LossFunction: #.##}");
        Console.WriteLine($"*\tR Squared: {modelMetrics.RSquared: #.##}");
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