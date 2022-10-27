using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;
using MulticlassClassification;
using MulticlassClassification.AppLogic;
using MulticlassClassification.AppLogic.Abstracts;
using MulticlassClassification.AppLogic.Implementations;
using MulticlassClassification.Model;

var predictor = new Predictor();

var newSample = new DiamondInput
{
    Carat = 1.1f,
    Color = "H",
    Clarity = "SI1",
    Polish = "VG",
    Symmetry = "EX",
    Report = "GIA",
    Price = 5169
};

Console.WriteLine("\n---------------------------------------Maximum Entropy Classification Trainers-------------------------------------------------");

var maxEntropyTrainers = new List<TrainerAbstract<MaximumEntropyModelParameters>>
{
    new LbfgsMaximumEntropyTrainer(),
    new SdcaMaximumEntropyTrainer()
};

maxEntropyTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------Naive Bayes Classification Trainers-------------------------------------------------");

var naiveTrainers = new List<TrainerAbstract<NaiveBayesMulticlassModelParameters>>
{
    new NaiveBayesTrainer()
};

naiveTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------One Versus All Classification Trainers-------------------------------------------------");

var oneVAllTrainers = new List<TrainerAbstract<OneVersusAllModelParameters>>
{
    new OneVersusAllClassificationTrainer()
};

oneVAllTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------Non Calibrated Classification Trainers-------------------------------------------------");

var nonCalibratedTrainers = new List<TrainerAbstract<LinearMulticlassModelParameters>>
{
    new SdcaNonCalibratedClassificationTrainer()
};

nonCalibratedTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n-----------------AutoML------------------\n");

//Load sample data
var sampleData = new MClassification.ModelInput()
{
    Carat_Weight = 0.83F,
    Color = @"H",
    Clarity = @"VS1",
    Polish = @"ID",
    Symmetry = @"ID",
    Report = @"AGSL",
    Price = 3470F,
};

//Load model and predict output
var result = MClassification.Predict(sampleData);

Console.WriteLine($"Cut: {result.Cut.ToString()}");