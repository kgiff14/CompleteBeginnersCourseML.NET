using Microsoft.ML.Data;

namespace DecisionTreeRegression.Models;

public class EnergyEfficiencyOutput
{
    [ColumnName("Score")]
    public float HeatingLoad { get; set; }
}