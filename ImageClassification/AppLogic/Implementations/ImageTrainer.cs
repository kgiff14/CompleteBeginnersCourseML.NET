using ImageClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Vision;

namespace ImageClassification.AppLogic.Implementations;

public class ImageTrainer : TrainerAbstract<ImageClassificationModelParameters>
{
    public ImageTrainer(ImageClassificationTrainer.Architecture architecture)
    {
        Model = MlContext.MulticlassClassification.Trainers.ImageClassification(new ImageClassificationTrainer.Options()
                                                                                    {
                                                                                        FeatureColumnName = "Features",
                                                                                        LabelColumnName = "Label",
                                                                                        ValidationSet = ValidationSet,
                                                                                        Arch = architecture,
                                                                                        MetricsCallback = (metrics) => Console.WriteLine(metrics),
                                                                                        TestOnTrainSet = false,
                                                                                        Epoch = 20
                                                                                    });
    }
}
