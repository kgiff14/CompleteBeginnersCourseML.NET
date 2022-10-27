using Microsoft.ML.Data;

namespace MulticlassClassification.Model;

public class DiamondInput
{
    [LoadColumn(0)]
    public float Carat { get; set; }

    [LoadColumn(1)]
    public string Label { get; set; }

    [LoadColumn(2)]
    public string Color { get; set; }

    [LoadColumn(3)]
    public string Clarity { get; set; }

    [LoadColumn(4)]
    public string Polish { get; set; }

    [LoadColumn(5)]
    public string Symmetry { get; set; }

    [LoadColumn(6)]
    public string Report { get; set; }

    [LoadColumn(7)]
    public float Price { get; set; }
}
