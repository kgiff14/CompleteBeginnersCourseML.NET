using Microsoft.ML;
using ObjectDetection.Models;
using System.Drawing;

namespace ObjectDetection.AppLogic;

public class Predictor
{
    private readonly MLContext _mLContext;
    private readonly PredictionEngine<ImageInput, PostProcessing> _predictionEngine;

    public Predictor(ITransformer trainedModel)
    {
        _mLContext = new(0);
        _predictionEngine = _mLContext.Model.CreatePredictionEngine<ImageInput, PostProcessing>(trainedModel);
    }

    public PostProcessing Predict(Bitmap image) =>
        _predictionEngine.Predict(new ImageInput() { Image = image });
}
