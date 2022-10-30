using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using Microsoft.ML;
using ImageClassification.Models;
using Microsoft.ML.Trainers;

namespace ImageClassification.AppLogic.Abstracts;

public class TrainerAbstract<T> where T : class
{
    public string ModelPath => Path.Combine(AppContext.BaseDirectory, "imageClassification.mdl");
    private string[] _trainingFolders => Directory.GetDirectories(Path.Combine(AppDomain.CurrentDomain.BaseDirectory.Split(@"\bin")[0], @"Assets\Sports"));

    protected ITrainerEstimator<MulticlassPredictionTransformer<T>, T> Model;
    protected MLContext MlContext;
    protected IDataView TrainSet;
    protected IDataView ValidationSet;
    protected IDataView TestSet;
    protected IEstimator<ITransformer> Pipeline;
    private ITransformer _trainedModel;

    public TrainerAbstract()
    {
        MlContext = new(0);

        Pipeline = CreateFeatureEngineerPipeline();
        CreateDataSplit(Pipeline);
    }

    public MulticlassClassificationMetrics Evaluate()
    {
        var testSet = _trainedModel.Transform(TestSet);

        return MlContext.MulticlassClassification.Evaluate(testSet);
    }

    public void Fit()
    {
        var modelPipeline = Pipeline.Append(Model)
                                    .Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")); ;

        _trainedModel = modelPipeline.Fit(TrainSet);
    }

    public void Save() =>
        MlContext.Model.Save(_trainedModel, TrainSet.Schema, ModelPath);

    protected IEstimator<ITransformer> CreateFeatureEngineerPipeline()
    {
        return MlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(ImageInput.Type), outputColumnName: "Label")
                    .Append(MlContext.Transforms.LoadRawImageBytes(outputColumnName: "Feature1", imageFolder: _trainingFolders[0], inputColumnName: nameof(ImageInput.ImagePath)))
                    .Append(MlContext.Transforms.LoadRawImageBytes(outputColumnName: "Feature2", imageFolder: _trainingFolders[1], inputColumnName: nameof(ImageInput.ImagePath)))
                    .Append(MlContext.Transforms.LoadRawImageBytes(outputColumnName: "Feature3", imageFolder: _trainingFolders[2], inputColumnName: nameof(ImageInput.ImagePath)))
                    .Append(MlContext.Transforms.LoadRawImageBytes(outputColumnName: "Feature4", imageFolder: _trainingFolders[3], inputColumnName: nameof(ImageInput.ImagePath)))
                    .Append(MlContext.Transforms.Concatenate("Features", new[]
                                                                        {
                                                                            "Feature1",
                                                                            "Feature2",
                                                                            "Feature3",
                                                                            "Feature4",
                                                                        }))
                    .AppendCacheCheckpoint(MlContext);
    }

    protected void CreateDataSplit(IEstimator<ITransformer> estimatorChain)
    {
        IEnumerable<ImageData> images = GetImageData();

        var imageData = MlContext.Data.LoadFromEnumerable(images);
        var shuffledImageData = MlContext.Data.ShuffleRows(imageData);

        var preparedData = estimatorChain.Fit(shuffledImageData).Transform(shuffledImageData);
        var trainSplit = MlContext.Data.TrainTestSplit(data: preparedData, testFraction: 0.2);
        var validationSplit = MlContext.Data.TrainTestSplit(trainSplit.TestSet);

        TrainSet = trainSplit.TrainSet;
        ValidationSet = validationSplit.TrainSet;
        TestSet = validationSplit.TestSet;
    }

    private IEnumerable<ImageData> GetImageData()
    {
        foreach (var folder in _trainingFolders)
        {
            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = folder.Split(@"\Sports\")[1];

                yield return new ImageData()
                {
                    ImagePath = file,
                    Type = label
                };
            }
        }
    }
}
