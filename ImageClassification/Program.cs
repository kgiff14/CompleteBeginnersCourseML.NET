using ImageClassification;
using ImageClassification.AppLogic;
using ImageClassification.AppLogic.Abstracts;
using ImageClassification.AppLogic.Implementations;
using ImageClassification.Models;
using Microsoft.ML.Vision;

var predictor = new Predictor();
ImageInput newSample;

var images =  Directory.GetFiles(Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Assets\Test"));

newSample = new ImageInput
{
    Image = File.ReadAllBytes(images[0])
};


Console.WriteLine("\n---------------------------------------Image Classification Trainers-------------------------------------------------");

var maxEntropyTrainers = new List<TrainerAbstract<ImageClassificationModelParameters>>
{
    new ImageTrainer(ImageClassificationTrainer.Architecture.ResnetV2101)
};



maxEntropyTrainers.ForEach(x => predictor.Predict(newSample, x));

//Load sample data
var imageBytes = File.ReadAllBytes(@"C:\Users\korde\source\repos\CompleteBeginnersCourseML.NET\ImageClassification\Assets\Sports\baseball\001.jpg");
ImageDetection.ModelInput sampleData = new()
{
    ImageSource = imageBytes,
};

//Load model and predict output
var result = ImageDetection.Predict(sampleData);

Console.WriteLine($"Prediction: {result.PredictedLabel}");
