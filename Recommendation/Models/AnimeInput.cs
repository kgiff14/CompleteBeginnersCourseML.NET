using Microsoft.ML.Data;

namespace Recommendation.Models;

public class AnimeInput
{
    [LoadColumn(0)]
    public float UserId { get; set; }

    [LoadColumn(1)]
    public float AnimeId { get; set; }

    [LoadColumn(2)]
    public float Rating { get; set; }
}
