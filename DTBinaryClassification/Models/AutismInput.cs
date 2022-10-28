using Microsoft.ML.Data;

namespace DTBinaryClassification.Models;

public class AutismInput
{
    [LoadColumn(0)]
    public float Id { get; set; }

    [LoadColumn(1)]
    public float A1 { get; set; }

    [LoadColumn(2)]
    public float A2 { get; set; }

    [LoadColumn(3)]
    public float A3 { get; set; }

    [LoadColumn(4)]
    public float A4 { get; set; }

    [LoadColumn(5)]
    public float A5 { get; set; }

    [LoadColumn(6)]
    public float A6 { get; set; }

    [LoadColumn(7)]
    public float A7 { get; set; }

    [LoadColumn(8)]
    public float A8 { get; set; }

    [LoadColumn(9)]
    public float A9 { get; set; }

    [LoadColumn(10)]
    public float A10 { get; set; }

    [LoadColumn(11)]
    public float Age { get; set; }

    [LoadColumn(12)]
    public string Gender { get; set; }

    [LoadColumn(13)]
    public string Ethnicity { get; set; }

    [LoadColumn(14)]
    public string Jundice { get; set; }

    [LoadColumn(15)]
    public string Autism { get; set; }

    [LoadColumn(16)]
    public string Country { get; set; }

    [LoadColumn(17)]
    public string UsedApp { get; set; }

    [LoadColumn(18)]
    public float Result { get; set; }

    [LoadColumn(19)]
    public float AgeDesc { get; set; }

    [LoadColumn(20)]
    public string Relation { get; set; }

    [LoadColumn(21)]
    public bool Class { get; set; }
}