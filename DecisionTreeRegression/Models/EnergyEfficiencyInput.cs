using Microsoft.ML.Data;

namespace DecisionTreeRegression.Models;

public class EnergyEfficiencyInput
{
    [LoadColumn(0)]
    public float Compactness { get; set; }

    [LoadColumn(1)]
    public float SurfaceArea { get; set; }

    [LoadColumn(2)]
    public float WallArea { get; set; }

    [LoadColumn(3)]
    public float RoofArea { get; set; }

    [LoadColumn(4)]
    public float Height { get; set; }

    [LoadColumn(5)]
    public float Orientation { get; set; }

    [LoadColumn(6)]
    public float GlazingArea { get; set; }

    [LoadColumn(7)]
    public float GlazingAreaDistribution { get; set; }

    [LoadColumn(8)]
    public float HeatingLoad { get; set; }
}
