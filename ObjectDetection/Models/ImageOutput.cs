namespace ObjectDetection.Models;

public class ImageOutput
{
    /// <summary>
    /// x1, y1, x2, y2 in page coordinates
    /// </summary>
    public float[] BoundingBox { get; set; }

    /// <summary>
    /// The Bounding box category
    /// </summary>
    public string Label { get; set; }

    /// <summary>
    /// The Confidence level
    /// </summary>
    public float Confidence { get; set; }

    public ImageOutput(float[] boundingBox, string label, float confidence)
    {
        BoundingBox = boundingBox;
        Label = label;
        Confidence = confidence;
    }
}
