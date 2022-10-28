using Microsoft.ML.Data;

namespace DTBinaryClassification.Models;

public class AutismOutput
{
    [ColumnName("PredictedLabel")]
    public bool ASD { get; set; }
}
