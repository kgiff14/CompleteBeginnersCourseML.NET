using Clustering.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace Clustering.AppLogic.Implementations;

public class KMeansClusterTrainer : TrainerAbstract<KMeansModelParameters>
{
    public KMeansClusterTrainer(int numberOfClusters) =>
        Model = MlContext.Clustering.Trainers.KMeans(numberOfClusters: numberOfClusters);
}
