using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using DecisionTree16Personality.Models;

namespace DecisionTree16Personality.AppLogic.Abstracts;

public abstract class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "decisionTreePersonality.mdl");

    protected MLContext MlContext;
    protected DataOperationsCatalog.TrainTestData DataSplit;
    protected ITrainerEstimator<MulticlassPredictionTransformer<T>, T> Model;
    private ITransformer _trainedModel;

    public TrainerAbstract()
    {
        MlContext = new(0);
    }

    public MulticlassClassificationMetrics Evaluate()
    {
        var testSet = _trainedModel.Transform(DataSplit.TestSet);

        return MlContext.MulticlassClassification.Evaluate(testSet);
    }

    public void Fit(string fileName)
    {
        CreateDataSplit(fileName);
        var pipeline = CreateFeatureEngineerPipeline();
        var modelPipeline = pipeline.Append(Model)
                                    .Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")); ;

        _trainedModel = modelPipeline.Fit(DataSplit.TrainSet);
    }

    public void Save() =>
        MlContext.Model.Save(_trainedModel, DataSplit.TrainSet.Schema, ModelPath);

    private void CreateDataSplit(string fileName)
    {
        if (!File.Exists(fileName))
            throw new FileNotFoundException($"{fileName} was not found");

        var dataSet = MlContext.Data.LoadFromTextFile<PersonalityInput>(fileName, hasHeader: true, separatorChar: ',');
        DataSplit = MlContext.Data.TrainTestSplit(dataSet, testFraction: 0.3);
    }

    private IEstimator<ITransformer> CreateFeatureEngineerPipeline()
    {
        var pipeline = MlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(PersonalityInput.Personality), outputColumnName: "Label")
                    .Append(MlContext.Transforms.Concatenate("Features", new[]
                                                                {
                                                                    nameof(PersonalityInput.Question1),
                                                                    nameof(PersonalityInput.Question2),
                                                                    nameof(PersonalityInput.Question3),
                                                                    nameof(PersonalityInput.Question4),
                                                                    nameof(PersonalityInput.Question5),
                                                                    nameof(PersonalityInput.Question6),
                                                                    nameof(PersonalityInput.Question7),
                                                                    nameof(PersonalityInput.Question8),
                                                                    nameof(PersonalityInput.Question9),
                                                                    nameof(PersonalityInput.Question10),
                                                                    nameof(PersonalityInput.Question11),
                                                                    nameof(PersonalityInput.Question12),
                                                                    nameof(PersonalityInput.Question13),
                                                                    nameof(PersonalityInput.Question14),
                                                                    nameof(PersonalityInput.Question15),
                                                                    nameof(PersonalityInput.Question16),
                                                                    nameof(PersonalityInput.Question17),
                                                                    nameof(PersonalityInput.Question18),
                                                                    nameof(PersonalityInput.Question19),
                                                                    nameof(PersonalityInput.Question20),
                                                                    nameof(PersonalityInput.Question21),
                                                                    nameof(PersonalityInput.Question22),
                                                                    nameof(PersonalityInput.Question23),
                                                                    nameof(PersonalityInput.Question24),
                                                                    nameof(PersonalityInput.Question25),
                                                                    nameof(PersonalityInput.Question26),
                                                                    nameof(PersonalityInput.Question27),
                                                                    nameof(PersonalityInput.Question28),
                                                                    nameof(PersonalityInput.Question29),
                                                                    nameof(PersonalityInput.Question30),
                                                                    nameof(PersonalityInput.Question31),
                                                                    nameof(PersonalityInput.Question32),
                                                                    nameof(PersonalityInput.Question33),
                                                                    nameof(PersonalityInput.Question34),
                                                                    nameof(PersonalityInput.Question35),
                                                                    nameof(PersonalityInput.Question36),
                                                                    nameof(PersonalityInput.Question37),
                                                                    nameof(PersonalityInput.Question38),
                                                                    nameof(PersonalityInput.Question39),
                                                                    nameof(PersonalityInput.Question40),
                                                                    nameof(PersonalityInput.Question41),
                                                                    nameof(PersonalityInput.Question42),
                                                                    nameof(PersonalityInput.Question43),
                                                                    nameof(PersonalityInput.Question44),
                                                                    nameof(PersonalityInput.Question45),
                                                                    nameof(PersonalityInput.Question46),
                                                                    nameof(PersonalityInput.Question47),
                                                                    nameof(PersonalityInput.Question48),
                                                                    nameof(PersonalityInput.Question49),
                                                                    nameof(PersonalityInput.Question50),
                                                                    nameof(PersonalityInput.Question51),
                                                                    nameof(PersonalityInput.Question52),
                                                                    nameof(PersonalityInput.Question53),
                                                                    nameof(PersonalityInput.Question54),
                                                                    nameof(PersonalityInput.Question55),
                                                                    nameof(PersonalityInput.Question56),
                                                                    nameof(PersonalityInput.Question57),
                                                                    nameof(PersonalityInput.Question58),
                                                                    nameof(PersonalityInput.Question59),
                                                                    nameof(PersonalityInput.Question60)
                                                                }));

        return pipeline;
    }
}
