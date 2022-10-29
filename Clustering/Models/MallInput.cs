using Microsoft.ML.Data;

namespace Clustering.Models;

public class MallInput
{
    [LoadColumn(0)]
    public float CustId { get; set; }

    [LoadColumn(1)]
    public string Gender { get; set; }

    [LoadColumn(2)]
    public float Age { get; set; }

    [LoadColumn(3)]
    public float Income { get; set; }

    [LoadColumn(4)]
    public float Spending { get; set; }
}