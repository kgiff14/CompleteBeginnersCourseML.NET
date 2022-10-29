using Microsoft.ML.Data;

namespace Forecasting.Models;

public class BitcoinOutput
{
    [ColumnName("Open")]
    public float[] Open { get; set; }

    [ColumnName("Open_LB")]
    public float[] Open_LB { get; set; }

    [ColumnName("Open_UB")]
    public float[] Open_UB { get; set; }
}
