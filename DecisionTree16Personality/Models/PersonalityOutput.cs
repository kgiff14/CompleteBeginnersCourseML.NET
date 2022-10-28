using Microsoft.ML.Data;

namespace DecisionTree16Personality.Models;

public class PersonalityOutput
{
    [ColumnName("PredictedLabel")]
    public string Personality { get; set; }
}
