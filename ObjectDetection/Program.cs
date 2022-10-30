using ObjectDetection.AppLogic;
using System.Drawing;

namespace ObjectDetection;

class Program
{
    private const string _modelPath = @"C:\Users\korde\source\repos\CompleteBeginnersCourseML.NET\ObjectDetection\Assets\Model\yolov4.onnx";
    private const string _imageFolder = @"C:\Users\korde\source\repos\CompleteBeginnersCourseML.NET\ObjectDetection\Assets\Images\";
    private const string _imageOutputFolder = @"C:\Users\korde\source\repos\CompleteBeginnersCourseML.NET\ObjectDetection\Assets\Output\";
    private static readonly string[] _classesNames = new string[] {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

    static void Main()
    {
        Directory.CreateDirectory(_imageOutputFolder);

        Console.WriteLine("Building and training yolov4 onnx model.");
        var trainer = new Trainer();
        var trainedModel = trainer.BuildAndTrain(_modelPath);

        var predictor = new Predictor(trainedModel);
        Console.WriteLine($"Creating predictions on images in: {_imageFolder}");
        DirectoryInfo directoryInfo = new(_imageFolder);
        FileInfo[] files = directoryInfo.GetFiles("*.jpg");

        foreach (FileInfo file in files)
        {
            using var image = new Bitmap(Image.FromFile(Path.Combine(_imageFolder, file.Name)));

            var postProcessing = predictor.Predict(image);
            var results = postProcessing.GetResults(_classesNames);

            BoundingBoxes.DrawAndStore(_imageOutputFolder, file.Name, results, image);
        }

        Console.WriteLine($"Check images in the output folder {_imageOutputFolder}");
    }
}