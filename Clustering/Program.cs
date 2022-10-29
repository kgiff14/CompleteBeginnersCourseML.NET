using Clustering.AppLogic;
using Clustering.AppLogic.Abstracts;
using Clustering.AppLogic.Implementations;
using Clustering.Models;
using Microsoft.ML.Trainers;

var predictor = new Predictor();

var newSample = new MallInput
{
    Gender = "Male",
    Age = 19f,
    Income = 15f,
    Spending = 39f

};

Console.WriteLine("Enter how many clusters do you desire");
var clusters = Convert.ToInt32(Console.ReadLine());

Console.WriteLine("\n---------------------------------------K Means Trainers-------------------------------------------------");

var kMeansTrainers = new List<TrainerAbstract<KMeansModelParameters>>
{
    new KMeansClusterTrainer(clusters)
};

kMeansTrainers.ForEach(x => predictor.Predict(newSample, x));