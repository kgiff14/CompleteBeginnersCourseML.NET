using Forecasting.AppLogic.Abstracts;
using Forecasting.Models;
using Microsoft.ML;

namespace Forecasting.AppLogic.Implementations;

public class ForecastTrainer : TrainerAbstract
{
    public ForecastTrainer() =>
        Model = MlContext.Forecasting.ForecastBySsa(windowSize: 10, seriesLength: 23, trainSize: 2788, horizon: 30, outputColumnName: "Open", inputColumnName: nameof(BitcoinInput.Open), confidenceLowerBoundColumn: "Open_LB", confidenceUpperBoundColumn: "Open_UB");
}
