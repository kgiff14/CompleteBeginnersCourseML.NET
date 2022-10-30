using AnomalyDetection.AppLogic;
using AnomalyDetection.AppLogic.Abstracts;
using AnomalyDetection.AppLogic.Implementations;
using AnomalyDetection.Models;
using Microsoft.ML.Trainers;

var predictor = new Predictor();

var newSample = new SkabInput
{
    Tempurature = 26.776f
};

Console.WriteLine("\n---------------------------------------Randomized PCA Trainers-------------------------------------------------");

var kMeansTrainers = new List<TrainerAbstract<PcaModelParameters>>
{
    new RandomizedPcaAnomalyTrainer()
};

kMeansTrainers.ForEach(x => predictor.Predict(newSample, x));