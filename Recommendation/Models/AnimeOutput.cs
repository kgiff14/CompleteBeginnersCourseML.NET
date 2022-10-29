using Microsoft.ML.Data;

namespace Recommendation.Models;

public class AnimeOutput
{
    [ColumnName("Score")]
    public float Score { get; set; }
}
