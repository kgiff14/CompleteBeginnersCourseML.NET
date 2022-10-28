﻿// This file was auto-generated by ML.NET Model Builder. 

using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML.Transforms.TimeSeries;

namespace Forecasting
{
    public partial class Forecast
    {
        /// <summary>
        /// model input class for Forecast.
        /// </summary>
        #region model input class
        public class ModelInput
        {
            [ColumnName(@"Open")]
            public float Open { get; set; }

        }

        #endregion

        /// <summary>
        /// model output class for Forecast.
        /// </summary>
        #region model output class
        public class ModelOutput
        {
            [ColumnName(@"Open")]
            public float[] Open { get; set; }

            [ColumnName(@"Open_LB")]
            public float[] Open_LB { get; set; }

            [ColumnName(@"Open_UB")]
            public float[] Open_UB { get; set; }

        }

        #endregion

        private static string MLNetModelPath = Path.GetFullPath(@"C:\Users\korde\source\repos\CompleteBeginnersCourseML.NET\Forecasting\Forecast.zip");

        public static readonly Lazy<TimeSeriesPredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<TimeSeriesPredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);

        /// <summary>
        /// Use this method to predict on <see cref="ModelInput"/>.
        /// </summary>
        /// <param name="input">model input.</param>
        /// <returns><seealso cref=" ModelOutput"/></returns>
        public static ModelOutput Predict(ModelInput? input = null, int? horizon = null)
        {
            var predEngine = PredictEngine.Value;
            return predEngine.Predict(input, horizon);
        }

        private static TimeSeriesPredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var schema);
            return mlModel.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
        }
    }
}

