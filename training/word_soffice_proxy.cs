using System;
using System.IO;
using System.Runtime.InteropServices;

internal static class Program
{
    public static int Main(string[] args)
    {
        string inputPath = null;
        string outputDirectory = null;

        for (int index = 0; index < args.Length; index++)
        {
            if (args[index] == "--outdir" && index + 1 < args.Length)
            {
                outputDirectory = args[++index];
            }
            else if (File.Exists(args[index]))
            {
                inputPath = args[index];
            }
        }

        if (inputPath == null || outputDirectory == null)
        {
            Console.Error.WriteLine("Word conversion proxy could not resolve input or output path.");
            return 2;
        }

        Directory.CreateDirectory(outputDirectory);
        string outputPath = Path.Combine(
            outputDirectory,
            Path.GetFileNameWithoutExtension(inputPath) + ".pdf"
        );

        object wordObject = null;
        object documentObject = null;
        try
        {
            Type wordType = Type.GetTypeFromProgID("Word.Application");
            if (wordType == null)
            {
                Console.Error.WriteLine("Microsoft Word is not available through COM.");
                return 3;
            }

            wordObject = Activator.CreateInstance(wordType);
            dynamic word = wordObject;
            word.Visible = false;
            word.DisplayAlerts = 0;
            documentObject = word.Documents.Open(
                Path.GetFullPath(inputPath),
                ConfirmConversions: false,
                ReadOnly: true,
                AddToRecentFiles: false,
                Visible: false
            );
            dynamic document = documentObject;
            document.ExportAsFixedFormat(outputPath, 17);
            document.Close(false);
            documentObject = null;
            word.Quit();
            wordObject = null;
            Console.WriteLine("Converted to " + outputPath);
            return File.Exists(outputPath) && new FileInfo(outputPath).Length > 0 ? 0 : 4;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 5;
        }
        finally
        {
            if (documentObject != null)
            {
                try { ((dynamic)documentObject).Close(false); } catch { }
                try { Marshal.FinalReleaseComObject(documentObject); } catch { }
            }
            if (wordObject != null)
            {
                try { ((dynamic)wordObject).Quit(); } catch { }
                try { Marshal.FinalReleaseComObject(wordObject); } catch { }
            }
        }
    }
}
