// This file was auto-generated by ML.NET Model Builder.
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
namespace DTBinaryClassification
{
    public partial class DTBinary
    {
        /// <summary>
        /// model input class for DTBinary.
        /// </summary>
        #region model input class
        public class ModelInput
        {
            [ColumnName(@"id")]
            public float Id { get; set; }

            [ColumnName(@"A1_Score")]
            public float A1_Score { get; set; }

            [ColumnName(@"A2_Score")]
            public float A2_Score { get; set; }

            [ColumnName(@"A3_Score")]
            public float A3_Score { get; set; }

            [ColumnName(@"A4_Score")]
            public float A4_Score { get; set; }

            [ColumnName(@"A5_Score")]
            public float A5_Score { get; set; }

            [ColumnName(@"A6_Score")]
            public float A6_Score { get; set; }

            [ColumnName(@"A7_Score")]
            public float A7_Score { get; set; }

            [ColumnName(@"A8_Score")]
            public float A8_Score { get; set; }

            [ColumnName(@"A9_Score")]
            public float A9_Score { get; set; }

            [ColumnName(@"A10_Score")]
            public float A10_Score { get; set; }

            [ColumnName(@"age")]
            public float Age { get; set; }

            [ColumnName(@"gender")]
            public string Gender { get; set; }

            [ColumnName(@"ethnicity")]
            public string Ethnicity { get; set; }

            [ColumnName(@"jundice")]
            public bool Jundice { get; set; }

            [ColumnName(@"austim")]
            public bool Austim { get; set; }

            [ColumnName(@"contry_of_res")]
            public string Contry_of_res { get; set; }

            [ColumnName(@"used_app_before")]
            public string Used_app_before { get; set; }

            [ColumnName(@"result")]
            public float Result { get; set; }

            [ColumnName(@"age_desc")]
            public string Age_desc { get; set; }

            [ColumnName(@"relation")]
            public string Relation { get; set; }

            [ColumnName(@"Class/ASD")]
            public bool Class_ASD { get; set; }

        }

        #endregion

        /// <summary>
        /// model output class for DTBinary.
        /// </summary>
        #region model output class
        public class ModelOutput
        {
            [ColumnName(@"id")]
            public float Id { get; set; }

            [ColumnName(@"A1_Score")]
            public float A1_Score { get; set; }

            [ColumnName(@"A2_Score")]
            public float A2_Score { get; set; }

            [ColumnName(@"A3_Score")]
            public float A3_Score { get; set; }

            [ColumnName(@"A4_Score")]
            public float A4_Score { get; set; }

            [ColumnName(@"A5_Score")]
            public float A5_Score { get; set; }

            [ColumnName(@"A6_Score")]
            public float A6_Score { get; set; }

            [ColumnName(@"A7_Score")]
            public float A7_Score { get; set; }

            [ColumnName(@"A8_Score")]
            public float A8_Score { get; set; }

            [ColumnName(@"A9_Score")]
            public float A9_Score { get; set; }

            [ColumnName(@"A10_Score")]
            public float A10_Score { get; set; }

            [ColumnName(@"age")]
            public float Age { get; set; }

            [ColumnName(@"gender")]
            public float[] Gender { get; set; }

            [ColumnName(@"ethnicity")]
            public float[] Ethnicity { get; set; }

            [ColumnName(@"jundice")]
            public float Jundice { get; set; }

            [ColumnName(@"austim")]
            public float Austim { get; set; }

            [ColumnName(@"contry_of_res")]
            public float[] Contry_of_res { get; set; }

            [ColumnName(@"used_app_before")]
            public string Used_app_before { get; set; }

            [ColumnName(@"result")]
            public float Result { get; set; }

            [ColumnName(@"age_desc")]
            public string Age_desc { get; set; }

            [ColumnName(@"relation")]
            public string Relation { get; set; }

            [ColumnName(@"Class/ASD")]
            public bool Class_ASD { get; set; }

            [ColumnName(@"Features")]
            public float[] Features { get; set; }

            [ColumnName(@"PredictedLabel")]
            public bool PredictedLabel { get; set; }

            [ColumnName(@"Score")]
            public float Score { get; set; }

            [ColumnName(@"Probability")]
            public float Probability { get; set; }

        }

        #endregion

        private static string MLNetModelPath = Path.GetFullPath("DTBinary.zip");

        public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);

        /// <summary>
        /// Use this method to predict on <see cref="ModelInput"/>.
        /// </summary>
        /// <param name="input">model input.</param>
        /// <returns><seealso cref=" ModelOutput"/></returns>
        public static ModelOutput Predict(ModelInput input)
        {
            var predEngine = PredictEngine.Value;
            return predEngine.Predict(input);
        }

        private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        }
    }
}
