using Recommendation;
using Recommendation.AppLogic;
using Recommendation.AppLogic.Abstracts;
using Recommendation.AppLogic.Implementations;
using Recommendation.Models;

var predictor = new Predictor();

var newSample = new AnimeInput
{
    UserId = 1f,
    AnimeId = 20f
};

Console.WriteLine("Enter how many approximation ranks you desire");
var ranks = Convert.ToInt32(Console.ReadLine());

Console.WriteLine("Enter how many training iterations you desire");
var iterations = Convert.ToInt32(Console.ReadLine());

Console.WriteLine("Enter the learning rate you desire");
var learningRate = Convert.ToDouble(Console.ReadLine());

Console.WriteLine("\n---------------------------------------Matrix Factorization Trainers-------------------------------------------------");

var fastTreeTrainers = new List<TrainerAbstract>
{
    new MatrixFactorizationTrainer(ranks, learningRate, iterations)
};

fastTreeTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n-----------------AutoML------------------\n");

//Load sample data
var sampleData = new Recommendations.ModelInput()
{
    User_id = 1F,
    Anime_id = 20F,
};

//Load model and predict output
var result = Recommendations.Predict(sampleData);

Console.WriteLine($"Rating: {result.Score}");