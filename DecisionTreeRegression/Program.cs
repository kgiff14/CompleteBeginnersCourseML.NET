using DecisionTreeRegression;
using DecisionTreeRegression.AppLogic;
using DecisionTreeRegression.AppLogic.Abstracts;
using DecisionTreeRegression.AppLogic.Implementations;
using DecisionTreeRegression.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

var predictor = new Predictor();

var newSample = new EnergyEfficiencyInput
{
    Compactness = 0.98f,
    SurfaceArea = 514.5f,
    WallArea = 294f,
    RoofArea = 110.25f,
    Height = 7f,
    Orientation = 2f,
    GlazingArea = 0f,
    GlazingAreaDistribution = 0f
};

Console.WriteLine("\n---------------------------------------Fast Tree Trainers-------------------------------------------------");

var fastTreeTrainers = new List<TrainerAbstract<FastTreeRegressionModelParameters>>
{
    new DecisionTreeTrainer(5, 10),
    new DecisionTreeTrainer(5, 10, 0.5),
    new DecisionTreeTrainer(10, 2),
    new DecisionTreeTrainer(10, 2, 0.3)
};

fastTreeTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------Fast Tree Tweedie Trainers-------------------------------------------------");

var fastTreeTweedieTrainers = new List<TrainerAbstract<FastTreeTweedieModelParameters>>
{
    new FastTreeTweedieRegressionTrainer(5, 10),
    new FastTreeTweedieRegressionTrainer(5, 10, 0.5),
    new FastTreeTweedieRegressionTrainer(10, 2),
    new FastTreeTweedieRegressionTrainer(10, 2, 0.3)
};

fastTreeTweedieTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------Gam Trainers-------------------------------------------------");

var gamTrainers = new List<TrainerAbstract<GamRegressionModelParameters>>
{
    new GamTrainer()
};

gamTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n-----------------AutoML------------------\n");

//Load sample data
var sampleData = new DTRegression.ModelInput()
{
    Relative_Compactness = 0.98F,
    Surface_Area = 514.5F,
    Wall_Area = 294F,
    Roof_Area = 110.25F,
    Overall_Height = 7F,
    Orientation = 2F,
    Glazing_Area = 0F,
    Glazing_Area_Distribution = 0F,
    Cooling_Load = 21.33F,
};

//Load model and predict output
var result = DTRegression.Predict(sampleData);

Console.WriteLine($"Heating Load: {result.Heating_Load}");