/*
Purpose: Downloads NFL play-by-play CSV for a single season and converts it to JSONL with acquisition metadata and logs.
Persists: Writes data/nfl/test-2025-pbp/raw/<season>/pbp.jsonl and data/nfl/test-2025-pbp/meta/{acquisition.json,sources.json,download.log}.
Security Risks: Downloads remote data via HTTPS; no credentials are handled.
*/

using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using Microsoft.VisualBasic.FileIO;

const int defaultSeason = 2025;
const string defaultOutputRoot = "data/nfl/test-2025-pbp";
const string defaultSourceTemplate = "https://github.com/nflverse/nflfastR-data/raw/master/play_by_play_{season}.csv.gz";

var options = ParseOptions(args);
if (!options.IsValid)
{
    PrintUsage();
    return 1;
}

var season = options.Season ?? defaultSeason;
var outputRoot = options.OutputRoot ?? defaultOutputRoot;
var sourceUrl = options.SourceUrl ?? defaultSourceTemplate.Replace("{season}", season.ToString(CultureInfo.InvariantCulture));

var rawSeasonDir = Path.Combine(outputRoot, "raw", season.ToString(CultureInfo.InvariantCulture));
var metaDir = Path.Combine(outputRoot, "meta");
var outputJsonlPath = Path.Combine(rawSeasonDir, "pbp.jsonl");
var tempJsonlPath = Path.Combine(rawSeasonDir, "pbp.jsonl.tmp");
var tempDownloadPath = Path.Combine(rawSeasonDir, $"play_by_play_{season}.csv.gz.tmp");
var logPath = Path.Combine(metaDir, "download.log");

Directory.CreateDirectory(rawSeasonDir);
Directory.CreateDirectory(metaDir);
using var log = new StreamWriter(logPath, append: true) { AutoFlush = true };

void Log(string message)
{
    var timestamp = DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture);
    var line = $"[{timestamp}] {message}";
    Console.WriteLine(line);
    log.WriteLine(line);
}

if (File.Exists(outputJsonlPath) && !options.Force)
{
    Log($"Output already exists at {outputJsonlPath}. Use --force to re-download.");
    return 0;
}

if (File.Exists(tempJsonlPath))
{
    File.Delete(tempJsonlPath);
    Log($"Deleted temp JSONL file at {tempJsonlPath}.");
}

if (File.Exists(tempDownloadPath))
{
    File.Delete(tempDownloadPath);
    Log($"Deleted temp download file at {tempDownloadPath}.");
}

Log($"Downloading season {season} from {sourceUrl}...");
using (var http = new HttpClient())
{
    HttpResponseMessage response;
    try
    {
        response = await http.GetAsync(sourceUrl, HttpCompletionOption.ResponseHeadersRead);
    }
    catch (HttpRequestException ex)
    {
        Log($"HTTP request failed for {sourceUrl}: {ex.Message}");
        return 2;
    }

    Log($"HTTP {(int)response.StatusCode} {response.ReasonPhrase} for {sourceUrl}.");
    if (!response.IsSuccessStatusCode)
    {
        Log("Download failed. Verify the URL or pass --source-url to a valid data file.");
        return 2;
    }

    await using var sourceStream = await response.Content.ReadAsStreamAsync();
    await using var targetStream = File.Create(tempDownloadPath);
    await sourceStream.CopyToAsync(targetStream);
}

Log("Converting CSV to JSONL...");
await using (var fileStream = File.OpenRead(tempDownloadPath))
await using (var gzip = new GZipStream(fileStream, CompressionMode.Decompress))
using (var writer = new StreamWriter(tempJsonlPath))
using (var parser = new TextFieldParser(gzip))
{
    parser.TextFieldType = FieldType.Delimited;
    parser.SetDelimiters(",");
    parser.HasFieldsEnclosedInQuotes = true;

    if (parser.EndOfData)
    {
        throw new InvalidDataException("CSV is empty.");
    }

    var headers = parser.ReadFields() ?? Array.Empty<string>();
    Log($"CSV header fields: {headers.Length}.");

    while (!parser.EndOfData)
    {
        var fields = parser.ReadFields() ?? Array.Empty<string>();
        var record = new Dictionary<string, string?>(headers.Length, StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < headers.Length; i++)
        {
            record[headers[i]] = i < fields.Length ? fields[i] : null;
        }

        var json = JsonSerializer.Serialize(record);
        await writer.WriteLineAsync(json);
    }
}

File.Move(tempJsonlPath, outputJsonlPath, overwrite: true);
File.Delete(tempDownloadPath);
Log($"Wrote JSONL output to {outputJsonlPath}.");
Log($"Removed temp download file at {tempDownloadPath}.");

var acquisition = new AcquisitionMetadata(
    Dataset: "NFL play-by-play",
    Season: season,
    RetrievedUtc: DateTimeOffset.UtcNow,
    SourceUrl: sourceUrl,
    Method: "Downloaded CSV via HTTPS and converted to JSONL per play.",
    Notes: "One-time snapshot for test data."
);

var sources = new SourceMetadata(
    Season: season,
    SourceUrl: sourceUrl,
    SourceFormat: "csv.gz",
    OutputFormat: "jsonl",
    OutputPath: outputJsonlPath
);

var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
await File.WriteAllTextAsync(Path.Combine(metaDir, "acquisition.json"), JsonSerializer.Serialize(acquisition, jsonOptions));
await File.WriteAllTextAsync(Path.Combine(metaDir, "sources.json"), JsonSerializer.Serialize(sources, jsonOptions));

Log($"Wrote metadata files to {metaDir}.");
Log("Done.");
return 0;

static Options ParseOptions(string[] args)
{
    var options = new Options();
    for (var i = 0; i < args.Length; i++)
    {
        var arg = args[i];
        switch (arg)
        {
            case "--season" when i + 1 < args.Length:
                if (int.TryParse(args[++i], NumberStyles.Integer, CultureInfo.InvariantCulture, out var season))
                {
                    options.Season = season;
                }
                else
                {
                    options.IsValid = false;
                }
                break;
            case "--output-root" when i + 1 < args.Length:
                options.OutputRoot = args[++i];
                break;
            case "--source-url" when i + 1 < args.Length:
                options.SourceUrl = args[++i];
                break;
            case "--force":
                options.Force = true;
                break;
            case "-h":
            case "--help":
                options.ShowHelp = true;
                break;
            default:
                options.IsValid = false;
                break;
        }
    }

    if (options.ShowHelp)
    {
        options.IsValid = false;
    }

    return options;
}

static void PrintUsage()
{
    Console.WriteLine("Usage: dotnet run -- [--season 2025] [--output-root path] [--source-url url] [--force]");
    Console.WriteLine($"Default season: {defaultSeason}");
    Console.WriteLine($"Default output root: {defaultOutputRoot}");
    Console.WriteLine("Default source URL: https://github.com/nflverse/nflfastR-data/raw/master/play_by_play_{season}.csv.gz");
    Console.WriteLine("Note: If the season URL does not exist, pass --source-url to a valid data file.");
}

sealed class Options
{
    public int? Season { get; set; }
    public string? OutputRoot { get; set; }
    public string? SourceUrl { get; set; }
    public bool Force { get; set; }
    public bool ShowHelp { get; set; }
    public bool IsValid { get; set; } = true;
}

sealed record AcquisitionMetadata(
    string Dataset,
    int Season,
    DateTimeOffset RetrievedUtc,
    string SourceUrl,
    string Method,
    string Notes
);

sealed record SourceMetadata(
    int Season,
    string SourceUrl,
    string SourceFormat,
    string OutputFormat,
    string OutputPath
);
