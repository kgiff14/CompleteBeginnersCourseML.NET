namespace ImageClassification.Models;

public class ImageOutput
{
    public string ImagePath { get; set; }

    public string Type { get; set; }

    public string PredictedLabel { get; set; }
}
