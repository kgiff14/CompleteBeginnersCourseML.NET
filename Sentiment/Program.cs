using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;
using Sentiment;
using Sentiment.AppLogic;
using Sentiment.AppLogic.Abstracts;
using Sentiment.AppLogic.Implementations;
using Sentiment.Models;

var predictor = new Predictor();

Console.WriteLine("Please write your review");
var text = Console.ReadLine();

var newSample = new ReviewInput
{
    Text = text
};

Console.WriteLine("\n---------------------------------------Calibrated Binary Classification Trainers-------------------------------------------------");

var calibratedTrainers = new List<TrainerAbstract<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>>
{
    new SentimentTrainer(),
};

calibratedTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n-----------------AutoML------------------\n");

//Load sample data
var sampleData = new SentimentAnalysis.ModelInput()
{
    Col0 = text,
};

//Load model and predict output
var result = SentimentAnalysis.Predict(sampleData);

Console.WriteLine($"Is Positive: {Convert.ToBoolean(result.PredictedLabel)}");