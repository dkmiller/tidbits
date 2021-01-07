using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

namespace Sharepoint.Upload
{
    record File(string Directory, string Name, FileInfo Info);

    record FilesystemClient(string Root, string Glob)
    {
        public IEnumerable<File> Search()
        {
            var rawFiles = Directory.GetFiles(Root, Glob, SearchOption.AllDirectories);

            // https://stackoverflow.com/a/146747
            var regex = new Regex(Regex.Escape(Root));

            foreach (var rawFile in rawFiles)
            {
                var relativePath = regex.Replace(rawFile, "", 1);
                var directory = Path.GetDirectoryName(relativePath).Replace(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                var name = Path.GetFileName(relativePath);
                var info = new FileInfo(rawFile);

                yield return new File(directory, name, info);
            }
        }
    }
}