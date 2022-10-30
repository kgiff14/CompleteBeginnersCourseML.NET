using Microsoft.ML.Data;

namespace Sentiment.Models;

public class ReviewOutput
{
    [ColumnName("PredictedLabel")]
    public bool IsPositive { get; set; }
}
