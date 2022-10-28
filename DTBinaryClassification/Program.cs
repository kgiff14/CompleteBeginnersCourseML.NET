using DTBinaryClassification;
using DTBinaryClassification.AppLogic;
using DTBinaryClassification.AppLogic.Abstracts;
using DTBinaryClassification.AppLogic.Implementations;
using DTBinaryClassification.Models;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

var predictor = new Predictor();

var newSample = new AutismInput
{
    Id = 3f,
    A1 = 1f,
    A2 = 1f,
    A3 = 1f,
    A4 = 1f,
    A5 = 0f,
    A6 = 0f,
    A7 = 1f,
    A8 = 1f,
    A9 = 0f,
    A10 = 0f,
    Age = 26f,
    Gender = "f",
    Ethnicity = "White-European",
    Jundice = "no",
    Autism = "no",
    Country = "United States",
    UsedApp = "no",
    Result = 6f,
    AgeDesc = 18f,
    Relation = "Self"
};

Console.WriteLine("Enter how many leaves you desire");
var leaves = Convert.ToInt32(Console.ReadLine());

Console.WriteLine("Enter how many trees you desire");
var trees = Convert.ToInt32(Console.ReadLine());

Console.WriteLine("Enter the learning rate you desire");
var learningRate = Convert.ToDouble(Console.ReadLine());

Console.WriteLine("\n---------------------------------------Fast Tree Trainers-------------------------------------------------");

var fastTreeTrainers = new List<TrainerAbstract<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>>
{
    new FastTreeTrainer(leaves, trees, learningRate)
};

fastTreeTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------Fast Forest Trainers-------------------------------------------------");

var fastForstTrainers = new List<TrainerAbstract<FastForestBinaryModelParameters>>
{
    new FastForestTrainer(leaves, trees)
};

fastForstTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------Gam Trainers-------------------------------------------------");

var gamTrainers = new List<TrainerAbstract<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>>
{
    new GamTrainer()
};

gamTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------Light Gbm Trainers-------------------------------------------------");

var gbmTrainers = new List<TrainerAbstract<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>>
{
    new LightGbmBinaryClassificationTrainer()
};

gbmTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n-----------------AutoML------------------\n");

// Create single instance of sample data from first line of dataset for model input
DTBinary.ModelInput sampleData = new DTBinary.ModelInput()
{
    Id = 4F,
    A1_Score = 1F,
    A2_Score = 1F,
    A3_Score = 0F,
    A4_Score = 1F,
    A5_Score = 0F,
    A6_Score = 0F,
    A7_Score = 0F,
    A8_Score = 1F,
    A9_Score = 0F,
    A10_Score = 1F,
    Age = 24F,
    Gender = @"m",
    Ethnicity = @"Latino",
    Contry_of_res = @"Brazil",
    Used_app_before = @"no",
    Result = 5F,
    Age_desc = @"18 and more",
    Relation = @"Self",
};

// Make a single prediction on the sample data and print results
var predictionResult = DTBinary.Predict(sampleData);

Console.WriteLine($"Is on ASD: {predictionResult.PredictedLabel}");