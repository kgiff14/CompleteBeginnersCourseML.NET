using BinaryClassification;
using BinaryClassification.AppLogic;
using BinaryClassification.AppLogic.Abstracts;
using BinaryClassification.AppLogic.Implementations;
using BinaryClassification.Models;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

var predictor = new Predictor();

var newSample = new BreastCancerInput
{
    Id = 842517F,
    RadiusMean = 20.57F,
    TextureMean = 17.77F,
    PerimeterMean = 132.9F,
    AreaMean = 1326F,
    SmoothnessMean = 0.08474F,
    CompactnessMean = 0.07864F,
    ConcavityMean = 0.0869F,
    ConcaveMean = 0.07017F,
    SymmetryMean = 0.1812F,
    FractialDimensionMean = 0.05667F,
    RadiusSe = 0.5435F,
    TextureSe = 0.7339F,
    PerimeterSe = 3.398F,
    AreaSe = 74.08F,
    SmoothnessSe = 0.005225F,
    CompactnessSe = 0.01308F,
    ConcavitySe = 0.0186F,
    ConcaveSe = 0.0134F,
    SymmetrySe = 0.01389F,
    FractialDimensionSe = 0.003532F,
    RadiusWorst = 24.99F,
    TextureWorst = 23.41F,
    PerimeterWorst = 158.8F,
    AreaWorst = 1956F,
    SmoothnessWorst = 0.1238F,
    CompactnessWorst = 0.1866F,
    ConcavityWorst = 0.2416F,
    ConcaveWorst = 0.186F,
    SymmetryWorst = 0.275F,
    FractialDimensionWorst = 0.08902F,
};

Console.WriteLine("\n---------------------------------------NonCalibrated Binary Classification Trainers-------------------------------------------------");

var nonCalibratedTrainers = new List<TrainerAbstract<LinearBinaryModelParameters>>
{
    new AveragedPerceptronClassificationTrainer(),
    new SdcaNonCalibratedTrainer(),
    new SgdNonCalibratedClassificationTrainer(),
};

nonCalibratedTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------Calibrated Binary Classification Trainers-------------------------------------------------");

var calibratedTrainers = new List<TrainerAbstract<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>>
{
    new LbfgsLogisticRegressionTrainer(),
    new SdcaLogisticRegressionTrainer(),
    new SgdCalibratedClassificationTrainer(),
    new SymbolicSgdLogisticRegressionTrainer()
};

calibratedTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n---------------------------------------Prior Binary Classification Trainers-------------------------------------------------");

var priorTrainers = new List<TrainerAbstract<PriorModelParameters>>
{
    new PriorClassificationTrainer()
};

priorTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n-----------------AutoML------------------\n");

//Load sample data
var sampleData = new BClassification.ModelInput()
{
    Id = 842517F,
    Radius_mean = 20.57F,
    Texture_mean = 17.77F,
    Perimeter_mean = 132.9F,
    Area_mean = 1326F,
    Smoothness_mean = 0.08474F,
    Compactness_mean = 0.07864F,
    Concavity_mean = 0.0869F,
    Concave_points_mean = 0.07017F,
    Symmetry_mean = 0.1812F,
    Fractal_dimension_mean = 0.05667F,
    Radius_se = 0.5435F,
    Texture_se = 0.7339F,
    Perimeter_se = 3.398F,
    Area_se = 74.08F,
    Smoothness_se = 0.005225F,
    Compactness_se = 0.01308F,
    Concavity_se = 0.0186F,
    Concave_points_se = 0.0134F,
    Symmetry_se = 0.01389F,
    Fractal_dimension_se = 0.003532F,
    Radius_worst = 24.99F,
    Texture_worst = 23.41F,
    Perimeter_worst = 158.8F,
    Area_worst = 1956F,
    Smoothness_worst = 0.1238F,
    Compactness_worst = 0.1866F,
    Concavity_worst = 0.2416F,
    Concave_points_worst = 0.186F,
    Symmetry_worst = 0.275F,
    Fractal_dimension_worst = 0.08902F,
};

//Load model and predict output
var result = BClassification.Predict(sampleData);

Console.WriteLine($"Is malignant: {Convert.ToBoolean(result.PredictedLabel)}");