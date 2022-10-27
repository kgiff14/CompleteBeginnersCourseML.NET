using Microsoft.ML.Data;

namespace BinaryClassification.Models;

public class BreastCancerOutput
{
    [ColumnName("PredictedLabel")]
    public bool IsMalignant { get; set; }
}
