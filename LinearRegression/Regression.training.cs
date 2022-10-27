﻿// This file was auto-generated by ML.NET Model Builder.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers;
using Microsoft.ML;

namespace LinearRegression
{
    public partial class Regression
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
            var pipeline = mlContext.Transforms.ReplaceMissingValues(new []{new InputOutputColumnPair(@"X1 transaction date", @"X1 transaction date"),new InputOutputColumnPair(@"X2 house age", @"X2 house age"),new InputOutputColumnPair(@"X3 distance to the nearest MRT station", @"X3 distance to the nearest MRT station"),new InputOutputColumnPair(@"X4 number of convenience stores", @"X4 number of convenience stores"),new InputOutputColumnPair(@"X5 latitude", @"X5 latitude"),new InputOutputColumnPair(@"X6 longitude", @"X6 longitude")})      
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"X1 transaction date",@"X2 house age",@"X3 distance to the nearest MRT station",@"X4 number of convenience stores",@"X5 latitude",@"X6 longitude"}))      
                                    .Append(mlContext.Regression.Trainers.FastForest(new FastForestRegressionTrainer.Options(){NumberOfTrees=10,NumberOfLeaves=4,FeatureFraction=0.8484519F,LabelColumnName=@"Y house price of unit area",FeatureColumnName=@"Features"}));

            return pipeline;
        }
    }
}
