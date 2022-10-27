using Microsoft.ML.Data;

namespace BinaryClassification.Models;

public class BreastCancerInput
{
    [LoadColumn(0)]
    public float Id { get; set; }

    [LoadColumn(1)]
    public bool Label { get; set; }

    [LoadColumn(2)]
    public float RadiusMean { get; set; }

    [LoadColumn(3)]
    public float TextureMean { get; set; }

    [LoadColumn(4)]
    public float PerimeterMean { get; set; }

    [LoadColumn(5)]
    public float AreaMean { get; set; }

    [LoadColumn(6)]
    public float SmoothnessMean { get; set; }

    [LoadColumn(7)]
    public float CompactnessMean { get; set; }

    [LoadColumn(8)]
    public float ConcavityMean { get; set; }

    [LoadColumn(9)]
    public float ConcaveMean { get; set; }

    [LoadColumn(10)]
    public float SymmetryMean { get; set; }

    [LoadColumn(11)]
    public float FractialDimensionMean { get; set; }

    [LoadColumn(12)]
    public float RadiusSe { get; set; }

    [LoadColumn(13)]
    public float TextureSe { get; set; }

    [LoadColumn(14)]
    public float PerimeterSe { get; set; }

    [LoadColumn(15)]
    public float AreaSe { get; set; }

    [LoadColumn(16)]
    public float SmoothnessSe { get; set; }

    [LoadColumn(17)]
    public float CompactnessSe { get; set; }

    [LoadColumn(18)]
    public float ConcavitySe { get; set; }

    [LoadColumn(19)]
    public float ConcaveSe { get; set; }

    [LoadColumn(20)]
    public float SymmetrySe { get; set; }

    [LoadColumn(21)]
    public float FractialDimensionSe { get; set; }

    [LoadColumn(22)]
    public float RadiusWorst { get; set; }

    [LoadColumn(23)]
    public float TextureWorst { get; set; }

    [LoadColumn(24)]
    public float PerimeterWorst { get; set; }

    [LoadColumn(25)]
    public float AreaWorst { get; set; }

    [LoadColumn(26)]
    public float SmoothnessWorst { get; set; }

    [LoadColumn(27)]
    public float CompactnessWorst { get; set; }

    [LoadColumn(28)]
    public float ConcavityWorst { get; set; }

    [LoadColumn(29)]
    public float ConcaveWorst { get; set; }

    [LoadColumn(30)]
    public float SymmetryWorst { get; set; }

    [LoadColumn(31)]
    public float FractialDimensionWorst { get; set; }
}