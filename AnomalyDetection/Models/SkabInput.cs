using Microsoft.ML.Data;

namespace AnomalyDetection.Models;

public class SkabInput
{
    [LoadColumn(0)]
    public string TimeStamp { get; set; }

    [LoadColumn(5)]
    public float Tempurature { get; set; }

    [LoadColumn(9)]
    public float Anomaly { get; set; }
}