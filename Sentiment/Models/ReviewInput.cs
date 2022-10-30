using Microsoft.ML.Data;

namespace Sentiment.Models;

public class ReviewInput
{
    [LoadColumn(0)]
    public string Text { get; set; }

    [LoadColumn(1)]
    public bool Label { get; set; }
}
