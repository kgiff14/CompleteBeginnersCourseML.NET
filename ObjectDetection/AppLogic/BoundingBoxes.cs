using ObjectDetection.Models;
using System.Drawing;

namespace ObjectDetection.AppLogic;

public static class BoundingBoxes
{
    public static void DrawAndStore(string imageOutputFolder, string imageName,
                    IReadOnlyList<ImageOutput> results, Bitmap image)
    {
        using (var graphics = Graphics.FromImage(image))
        {
            foreach (var result in results)
            {
                var x1 = result.BoundingBox[0];
                var y1 = result.BoundingBox[1];
                var x2 = result.BoundingBox[2];
                var y2 = result.BoundingBox[3];

                graphics.DrawRectangle(Pens.LightBlue, x1, y1, x2 - x1, y2 - y1);

                using (var brushes = new SolidBrush(Color.FromArgb(50, Color.LightBlue)))
                {
                    graphics.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                }

                graphics.DrawString(result.Label + " " + result.Confidence.ToString("0.00"), new Font("Open Sans", 12), Brushes.Black, new PointF(x1, y1));
            }

            image.Save(Path.Combine(imageOutputFolder, Path.ChangeExtension(imageName, "_result" + Path.GetExtension(imageName))));
        }
    }
}
