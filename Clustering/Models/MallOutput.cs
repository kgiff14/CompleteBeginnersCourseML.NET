using Microsoft.ML.Data;

namespace Clustering.Models;

public class MallOutput
{
    [ColumnName("Score")]
    public float[] Scores { get; set; }

    [ColumnName("PredictedLabel")]
    public UInt32 Label { get; set; }
}
