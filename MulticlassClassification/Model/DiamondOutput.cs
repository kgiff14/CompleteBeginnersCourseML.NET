using Microsoft.ML.Data;

namespace MulticlassClassification.Model;

public class DiamondOutput
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; }
}
