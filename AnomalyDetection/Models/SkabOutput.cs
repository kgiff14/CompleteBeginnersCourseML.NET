using Microsoft.ML.Data;

namespace AnomalyDetection.Models;

public class SkabOutput
{
    [ColumnName("PredictedLabel")]
    public bool IsAnomaly { get; set; }
}
