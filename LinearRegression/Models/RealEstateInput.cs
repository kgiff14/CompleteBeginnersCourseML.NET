using Microsoft.ML.Data;

namespace LinearRegression.Models;

public class RealEstateInput
{
    [LoadColumn(0)]
    public float Id { get; set; }
    [LoadColumn(1)]
    public float TransactionDate { get; set; }

    [LoadColumn(2)]
    public float HouseAge { get; set; }

    [LoadColumn(3)]
    public float NearestMRT { get; set; }

    [LoadColumn(4)]
    public float NumberOfStores { get; set; }

    [LoadColumn(5)]
    public float Latitude { get; set; }

    [LoadColumn(6)]
    public float Longitude { get; set; }

    [LoadColumn(7)]
    public float HousePrice { get; set; }
}
