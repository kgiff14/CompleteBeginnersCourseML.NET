namespace ImageClassification.Models;

public class ImageInput
{
    public byte[] Image { get; set; }

    public UInt32 Label { get; set; }

    public string ImagePath { get; set; }

    public string Type { get; set; }
}
