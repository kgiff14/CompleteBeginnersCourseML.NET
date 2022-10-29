using Microsoft.ML;
using Microsoft.ML.Trainers.Recommender;
using Recommendation.AppLogic.Abstracts;

namespace Recommendation.AppLogic.Implementations;

public class MatrixFactorizationTrainer : TrainerAbstract
{
    public MatrixFactorizationTrainer(int approximationRank, double learningRate, int numberOfIterations) =>
        Model = MlContext.Recommendation().Trainers.MatrixFactorization(labelColumnName: "Label", 
                                                                        matrixColumnIndexColumnName: "UserFeature", 
                                                                        matrixRowIndexColumnName: "AnimeFeature", 
                                                                        approximationRank: approximationRank, 
                                                                        learningRate: learningRate, 
                                                                        numberOfIterations: numberOfIterations);
}
