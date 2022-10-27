﻿// This file was auto-generated by ML.NET Model Builder.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;

namespace BinaryClassification
{
    public partial class BClassification
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
            var pipeline = mlContext.Transforms.ReplaceMissingValues(new []{new InputOutputColumnPair(@"radius_mean", @"radius_mean"),new InputOutputColumnPair(@"texture_mean", @"texture_mean"),new InputOutputColumnPair(@"perimeter_mean", @"perimeter_mean"),new InputOutputColumnPair(@"area_mean", @"area_mean"),new InputOutputColumnPair(@"smoothness_mean", @"smoothness_mean"),new InputOutputColumnPair(@"compactness_mean", @"compactness_mean"),new InputOutputColumnPair(@"concavity_mean", @"concavity_mean"),new InputOutputColumnPair(@"concave points_mean", @"concave points_mean"),new InputOutputColumnPair(@"symmetry_mean", @"symmetry_mean"),new InputOutputColumnPair(@"fractal_dimension_mean", @"fractal_dimension_mean"),new InputOutputColumnPair(@"radius_se", @"radius_se"),new InputOutputColumnPair(@"texture_se", @"texture_se"),new InputOutputColumnPair(@"perimeter_se", @"perimeter_se"),new InputOutputColumnPair(@"area_se", @"area_se"),new InputOutputColumnPair(@"smoothness_se", @"smoothness_se"),new InputOutputColumnPair(@"compactness_se", @"compactness_se"),new InputOutputColumnPair(@"concavity_se", @"concavity_se"),new InputOutputColumnPair(@"concave points_se", @"concave points_se"),new InputOutputColumnPair(@"symmetry_se", @"symmetry_se"),new InputOutputColumnPair(@"fractal_dimension_se", @"fractal_dimension_se"),new InputOutputColumnPair(@"radius_worst", @"radius_worst"),new InputOutputColumnPair(@"texture_worst", @"texture_worst"),new InputOutputColumnPair(@"perimeter_worst", @"perimeter_worst"),new InputOutputColumnPair(@"area_worst", @"area_worst"),new InputOutputColumnPair(@"smoothness_worst", @"smoothness_worst"),new InputOutputColumnPair(@"compactness_worst", @"compactness_worst"),new InputOutputColumnPair(@"concavity_worst", @"concavity_worst"),new InputOutputColumnPair(@"concave points_worst", @"concave points_worst"),new InputOutputColumnPair(@"symmetry_worst", @"symmetry_worst"),new InputOutputColumnPair(@"fractal_dimension_worst", @"fractal_dimension_worst")})      
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"radius_mean",@"texture_mean",@"perimeter_mean",@"area_mean",@"smoothness_mean",@"compactness_mean",@"concavity_mean",@"concave points_mean",@"symmetry_mean",@"fractal_dimension_mean",@"radius_se",@"texture_se",@"perimeter_se",@"area_se",@"smoothness_se",@"compactness_se",@"concavity_se",@"concave points_se",@"symmetry_se",@"fractal_dimension_se",@"radius_worst",@"texture_worst",@"perimeter_worst",@"area_worst",@"smoothness_worst",@"compactness_worst",@"concavity_worst",@"concave points_worst",@"symmetry_worst",@"fractal_dimension_worst"}))      
                                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName:@"diagnosis",inputColumnName:@"diagnosis"))      
                                    .Append(mlContext.Transforms.NormalizeMinMax(@"Features", @"Features"))      
                                    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(new LbfgsMaximumEntropyMulticlassTrainer.Options(){L1Regularization=1F,L2Regularization=1F,LabelColumnName=@"diagnosis",FeatureColumnName=@"Features"}))      
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName:@"PredictedLabel",inputColumnName:@"PredictedLabel"));

            return pipeline;
        }
    }
}
