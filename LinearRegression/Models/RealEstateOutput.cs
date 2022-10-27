using Microsoft.ML.Data;

namespace LinearRegression.Models;

public class RealEstateOutput
{
    [ColumnName("Score")]
    public float Price { get; set; }
}
