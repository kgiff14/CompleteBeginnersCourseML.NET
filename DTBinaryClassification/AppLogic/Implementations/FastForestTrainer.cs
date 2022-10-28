using DTBinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace DTBinaryClassification.AppLogic.Implementations;

public class FastForestTrainer : TrainerAbstract<FastForestBinaryModelParameters>
{
    public FastForestTrainer(int numberOfTrees, int numberOfLeaves) =>
        Model = MlContext.BinaryClassification.Trainers.FastForest(numberOfTrees: numberOfTrees, numberOfLeaves: numberOfLeaves);
}
