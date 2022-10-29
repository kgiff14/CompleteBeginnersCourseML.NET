using Microsoft.ML.Data;

namespace Forecasting.Models;

public class BitcoinInput
{
    [LoadColumn(1)]
    public float Open { get; set; }
}