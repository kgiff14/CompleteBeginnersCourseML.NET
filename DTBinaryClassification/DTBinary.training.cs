// This file was auto-generated by ML.NET Model Builder.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;

namespace DTBinaryClassification
{
    public partial class DTBinary
    {
        /// <summary>
        /// Retrains model using the pipeline generated as part of the training process. For more information on how to load data, see aka.ms/loaddata.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainData"></param>
        /// <returns></returns>
        public static ITransformer RetrainPipeline(MLContext mlContext, IDataView trainData)
        {
            var pipeline = BuildPipeline(mlContext);
            var model = pipeline.Fit(trainData);

            return model;
        }

        /// <summary>
        /// build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.ReplaceMissingValues(new []{new InputOutputColumnPair(@"A1_Score", @"A1_Score"),new InputOutputColumnPair(@"A2_Score", @"A2_Score"),new InputOutputColumnPair(@"A3_Score", @"A3_Score"),new InputOutputColumnPair(@"A4_Score", @"A4_Score"),new InputOutputColumnPair(@"A5_Score", @"A5_Score"),new InputOutputColumnPair(@"A6_Score", @"A6_Score"),new InputOutputColumnPair(@"A7_Score", @"A7_Score"),new InputOutputColumnPair(@"A8_Score", @"A8_Score"),new InputOutputColumnPair(@"A9_Score", @"A9_Score"),new InputOutputColumnPair(@"A10_Score", @"A10_Score"),new InputOutputColumnPair(@"age", @"age")})      
                                    .Append(mlContext.Transforms.Conversion.ConvertType(new []{new InputOutputColumnPair(@"jundice", @"jundice"),new InputOutputColumnPair(@"austim", @"austim")}))      
                                    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName:@"gender",outputColumnName:@"gender"))      
                                    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName:@"ethnicity",outputColumnName:@"ethnicity"))      
                                    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName:@"contry_of_res",outputColumnName:@"contry_of_res"))      
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"A1_Score",@"A2_Score",@"A3_Score",@"A4_Score",@"A5_Score",@"A6_Score",@"A7_Score",@"A8_Score",@"A9_Score",@"A10_Score",@"age",@"jundice",@"austim",@"gender",@"ethnicity",@"contry_of_res"}))      
                                    .Append(mlContext.Transforms.NormalizeMinMax(@"Features", @"Features"))      
                                    .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(new LbfgsLogisticRegressionBinaryTrainer.Options(){L1Regularization=1F,L2Regularization=1F,LabelColumnName=@"Class/ASD",FeatureColumnName=@"Features"}));

            return pipeline;
        }
    }
}
