using AnomalyDetection.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace AnomalyDetection.AppLogic.Implementations;

public class RandomizedPcaAnomalyTrainer : TrainerAbstract<PcaModelParameters>
{
    public RandomizedPcaAnomalyTrainer() =>
        Model = MlContext.AnomalyDetection.Trainers.RandomizedPca(new RandomizedPcaTrainer.Options()
                                                                        {
                                                                            FeatureColumnName = "Features",
                                                                            Rank = 1,
                                                                            Seed = 10,
                                                                        });
}
