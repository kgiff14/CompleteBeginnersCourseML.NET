using LinearRegression;
using LinearRegression.AppLogic;
using LinearRegression.AppLogic.Implementations;
using LinearRegression.AppLogic.Interfaces;
using LinearRegression.Models;
using Microsoft.ML;
using Microsoft.ML.Trainers;

var predictor = new Predictor();

var newSample = new RealEstateInput
{
    Id = 1f,
    TransactionDate = 2012.917f,
    HouseAge = 32f,
    NearestMRT = 84.87882f,
    NumberOfStores = 10f,
    Latitude = 24.98298f,
    Longitude = 121.54024f
};

Console.WriteLine("\nLinear Regression Trainers-------------------------------------------------");

var linearRegressionTrainers = new List<TrainerAbstract<LinearRegressionModelParameters>>
{
    new SdcaLinearRegressionTrainer(),
    new OnlineGradientDescentRegressionTrainer(),
};

linearRegressionTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\nPoisson Regression Trainers-------------------------------------------------");

var poissonRegressionTrainers = new List<TrainerAbstract<PoissonRegressionModelParameters>>
{
    new LbfgsPoissonLinearRegressionTrainer()
};

poissonRegressionTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n-----------------AutoML------------------\n");

//Load sample data
var sampleData = new Regression.ModelInput()
{
    No = 2F,
    X1_transaction_date = 2012.917F,
    X2_house_age = 19.5F,
    X3_distance_to_the_nearest_MRT_station = 306.5947F,
    X4_number_of_convenience_stores = 9F,
    X5_latitude = 24.98034F,
    X6_longitude = 121.5395F,
};

//Load model and predict output
var result = Regression.Predict(sampleData);

Console.WriteLine($"Prediction: {result.Score}");