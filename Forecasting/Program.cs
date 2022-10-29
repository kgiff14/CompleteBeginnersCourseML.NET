using Forecasting;
using Forecasting.AppLogic;
using Forecasting.AppLogic.Abstracts;
using Forecasting.AppLogic.Implementations;

var predictor = new Predictor();

Console.WriteLine("\n-------------------------------------------------Forecast Trainers-------------------------------------------------");

var linearRegressionTrainers = new List<TrainerAbstract>
{
    new ForecastTrainer()
};

linearRegressionTrainers.ForEach(x => predictor.Predict(x));

Console.WriteLine("\n-----------------AutoML------------------\n");

// Load model and predict the next set values.
// The number of values predicted is equal to the horizon specified while training.
var result = Forecast.Predict();


for(int i = 0; i < result.Open.Length; i++)
{
    Console.WriteLine($"Predictions: {result.Open[i]}");
}