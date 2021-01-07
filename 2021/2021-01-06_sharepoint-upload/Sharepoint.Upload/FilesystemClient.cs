using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace Sharepoint.Upload
{
    record File(string Directory, string Name, FileInfo Info);

    record FilesystemClient(string Root, string Glob)
    {
        public async IAsyncEnumerable<File> SearchAsync()
        {
            var rawFiles = Directory.GetFiles(Root, Glob, SearchOption.AllDirectories);

            foreach (var rawFile in rawFiles)
            {
                var relativePath = rawFile.Replace(Root, "");
                var directory = Path.GetDirectoryName(relativePath).Replace(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                var name = Path.GetFileName(relativePath);
                var info = new FileInfo(rawFile);

                yield return await Task.FromResult(new File(directory, name, info));
            }
        }
    }
}