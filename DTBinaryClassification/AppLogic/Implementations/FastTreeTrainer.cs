using DTBinaryClassification.AppLogic.Abstracts;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace DTBinaryClassification.AppLogic.Implementations;

public class FastTreeTrainer : TrainerAbstract<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>
{
    public FastTreeTrainer(int numberOfTrees, int numberOfLeaves, double learningRate = 0.2) =>
        Model = MlContext.BinaryClassification.Trainers.FastTree(numberOfTrees: numberOfTrees, numberOfLeaves: numberOfLeaves, learningRate: learningRate);
}
