// Purpose: Run a deterministic, debugger-friendly toy experiment comparing baseline prose prompts with CE scaffold formats while mixing formats in one model.
// Persists: None.
// Security Risks: None.

using System.Globalization;

namespace ToyConceptScaffold;

internal static class Program
{
    private const string Version = "v0.1";

    private enum PromptFormat
    {
        Baseline = 0,
        CeNeo = 1,
        CeEng = 2
    }

    private enum Label
    {
        Ok = 0,
        Slip = 1,
        Squish = 2
    }

    private enum Cause
    {
        None = 0,
        ShearLow = 1,
        FrictionDrop = 2,
        SlipOnset = 3,
        ComplianceHigh = 4
    }

    private sealed record Options(
        int Seed,
        int TrainSamples,
        int TestSamples,
        int Epochs,
        int EmbedDim,
        List<int> PadLevels,
        bool EnableParaphrase,
        bool PrintDemo,
        string Mode);

    private sealed record Example(int[] Tokens, Label Label, Cause Cause, LatentState Latent, string RawText);

    private sealed record LatentState(
        int SlipOnset,
        int ShearLow,
        int FrictionDrop,
        int ComplianceHigh);

    private sealed record ForwardResult(
        double[] Pooled,
        double[] LabelLogits,
        double[] CauseLogits,
        double[] LabelProbs,
        double[] CauseProbs);

    // Simple bag-of-tokens model:
    // 1) Sum embeddings into a pooled vector.
    // 2) Two linear heads predict label and cause.
    // This is intentionally tiny so you can step through every op in a debugger.
    private sealed class ToyModel
    {
        private const double GradClipNorm = 1.0;
        private readonly int _vocabSize;
        private readonly int _embedDim;
        private readonly double[,] _embeddings;
        private readonly double[,] _labelWeights;
        private readonly double[] _labelBias;
        private readonly double[,] _causeWeights;
        private readonly double[] _causeBias;
        private readonly Random _rng;

        public ToyModel(int vocabSize, int embedDim, int seed)
        {
            _vocabSize = vocabSize;
            _embedDim = embedDim;
            _embeddings = new double[vocabSize, embedDim];
            _labelWeights = new double[embedDim, Enum.GetValues<Label>().Length];
            _labelBias = new double[Enum.GetValues<Label>().Length];
            _causeWeights = new double[embedDim, Enum.GetValues<Cause>().Length];
            _causeBias = new double[Enum.GetValues<Cause>().Length];
            _rng = new Random(seed);
            InitWeights();
        }

        private void InitWeights()
        {
            // Small uniform init helps keep training stable and deterministic.
            const double scale = 0.01;
            for (var v = 0; v < _vocabSize; v++)
            {
                for (var d = 0; d < _embedDim; d++)
                {
                    _embeddings[v, d] = (NextUniform() * 2.0 - 1.0) * scale;
                }
            }

            for (var d = 0; d < _embedDim; d++)
            {
                for (var c = 0; c < _labelBias.Length; c++)
                {
                    _labelWeights[d, c] = (NextUniform() * 2.0 - 1.0) * scale;
                }

                for (var c = 0; c < _causeBias.Length; c++)
                {
                    _causeWeights[d, c] = (NextUniform() * 2.0 - 1.0) * scale;
                }
            }
        }

        // Forward pass that exposes pooled state + logits/probabilities for debugging.
        public ForwardResult Forward(int[] tokens)
        {
            var pooled = new double[_embedDim];
            for (var i = 0; i < tokens.Length; i++)
            {
                var token = tokens[i];
                for (var d = 0; d < _embedDim; d++)
                {
                    pooled[d] += _embeddings[token, d];
                }
            }

            var invTokenCount = 1.0 / Math.Max(1, tokens.Length);
            for (var d = 0; d < _embedDim; d++)
            {
                pooled[d] *= invTokenCount;
            }

            var labelLogits = new double[_labelBias.Length];
            var causeLogits = new double[_causeBias.Length];

            for (var c = 0; c < labelLogits.Length; c++)
            {
                var sum = _labelBias[c];
                for (var d = 0; d < _embedDim; d++)
                {
                    sum += pooled[d] * _labelWeights[d, c];
                }
                labelLogits[c] = sum;
            }

            for (var c = 0; c < causeLogits.Length; c++)
            {
                var sum = _causeBias[c];
                for (var d = 0; d < _embedDim; d++)
                {
                    sum += pooled[d] * _causeWeights[d, c];
                }
                causeLogits[c] = sum;
            }

            var labelProbs = Softmax(labelLogits);
            var causeProbs = Softmax(causeLogits);
            return new ForwardResult(pooled, labelLogits, causeLogits, labelProbs, causeProbs);
        }

        public (Label Label, Cause Cause) Predict(int[] tokens)
        {
            var forward = Forward(tokens);
            return ((Label)ArgMax(forward.LabelProbs), (Cause)ArgMax(forward.CauseProbs));
        }

        // One SGD step; gradients are derived directly from cross-entropy on softmax outputs.
        public double TrainStep(int[] tokens, Label label, Cause cause, double learningRate, int epoch, int sampleIndex)
        {
            var forward = Forward(tokens);

            var labelGrad = (double[])forward.LabelProbs.Clone();
            labelGrad[(int)label] -= 1.0;

            var causeGrad = (double[])forward.CauseProbs.Clone();
            causeGrad[(int)cause] -= 1.0;

            var gradPooled = new double[_embedDim];

            var dLabelWeights = new double[_embedDim, labelGrad.Length];
            var dCauseWeights = new double[_embedDim, causeGrad.Length];
            var dLabelBias = (double[])labelGrad.Clone();
            var dCauseBias = (double[])causeGrad.Clone();

            for (var d = 0; d < _embedDim; d++)
            {
                var grad = 0.0;
                for (var c = 0; c < labelGrad.Length; c++)
                {
                    grad += _labelWeights[d, c] * labelGrad[c];
                    dLabelWeights[d, c] = forward.Pooled[d] * labelGrad[c];
                }

                for (var c = 0; c < causeGrad.Length; c++)
                {
                    grad += _causeWeights[d, c] * causeGrad[c];
                    dCauseWeights[d, c] = forward.Pooled[d] * causeGrad[c];
                }

                gradPooled[d] = grad;
            }

            var invTokenCount = 1.0 / Math.Max(1, tokens.Length);
            var embeddingGrads = new Dictionary<int, double[]>();
            for (var i = 0; i < tokens.Length; i++)
            {
                var token = tokens[i];
                if (!embeddingGrads.TryGetValue(token, out var grad))
                {
                    grad = new double[_embedDim];
                    embeddingGrads[token] = grad;
                }

                for (var d = 0; d < _embedDim; d++)
                {
                    grad[d] += gradPooled[d] * invTokenCount;
                }
            }

            var labelLoss = -Math.Log(Math.Max(forward.LabelProbs[(int)label], 1e-12));
            var causeLoss = -Math.Log(Math.Max(forward.CauseProbs[(int)cause], 1e-12));
            var loss = labelLoss + causeLoss;

            if (double.IsNaN(loss) || double.IsInfinity(loss))
            {
                var maxPooled = MaxAbs(forward.Pooled);
                var maxLabelLogit = MaxAbs(forward.LabelLogits);
                var maxCauseLogit = MaxAbs(forward.CauseLogits);
                Console.WriteLine($"NaN loss @ epoch {epoch} sample {sampleIndex} max|h|={maxPooled:0.0000} max|labelLogit|={maxLabelLogit:0.0000} max|causeLogit|={maxCauseLogit:0.0000}");
                return 50.0;
            }

            ClipGradients(dLabelWeights, dCauseWeights, dLabelBias, dCauseBias, embeddingGrads, GradClipNorm);

            for (var d = 0; d < _embedDim; d++)
            {
                for (var c = 0; c < labelGrad.Length; c++)
                {
                    _labelWeights[d, c] -= learningRate * dLabelWeights[d, c];
                }

                for (var c = 0; c < causeGrad.Length; c++)
                {
                    _causeWeights[d, c] -= learningRate * dCauseWeights[d, c];
                }
            }

            for (var c = 0; c < labelGrad.Length; c++)
            {
                _labelBias[c] -= learningRate * dLabelBias[c];
            }

            for (var c = 0; c < causeGrad.Length; c++)
            {
                _causeBias[c] -= learningRate * dCauseBias[c];
            }

            foreach (var (token, grad) in embeddingGrads)
            {
                for (var d = 0; d < _embedDim; d++)
                {
                    _embeddings[token, d] -= learningRate * grad[d];
                }
            }

            return loss;
        }

        private double NextUniform() => _rng.NextDouble();

        private static void ClipGradients(
            double[,] dLabelWeights,
            double[,] dCauseWeights,
            double[] dLabelBias,
            double[] dCauseBias,
            Dictionary<int, double[]> embeddingGrads,
            double clipNorm)
        {
            var sumSquares = 0.0;
            for (var i = 0; i < dLabelWeights.GetLength(0); i++)
            {
                for (var j = 0; j < dLabelWeights.GetLength(1); j++)
                {
                    sumSquares += dLabelWeights[i, j] * dLabelWeights[i, j];
                }
            }

            for (var i = 0; i < dCauseWeights.GetLength(0); i++)
            {
                for (var j = 0; j < dCauseWeights.GetLength(1); j++)
                {
                    sumSquares += dCauseWeights[i, j] * dCauseWeights[i, j];
                }
            }

            for (var i = 0; i < dLabelBias.Length; i++)
            {
                sumSquares += dLabelBias[i] * dLabelBias[i];
            }

            for (var i = 0; i < dCauseBias.Length; i++)
            {
                sumSquares += dCauseBias[i] * dCauseBias[i];
            }

            foreach (var grad in embeddingGrads.Values)
            {
                for (var i = 0; i < grad.Length; i++)
                {
                    sumSquares += grad[i] * grad[i];
                }
            }

            var norm = Math.Sqrt(sumSquares);
            if (norm <= clipNorm || norm == 0.0)
            {
                return;
            }

            var scale = clipNorm / norm;
            for (var i = 0; i < dLabelWeights.GetLength(0); i++)
            {
                for (var j = 0; j < dLabelWeights.GetLength(1); j++)
                {
                    dLabelWeights[i, j] *= scale;
                }
            }

            for (var i = 0; i < dCauseWeights.GetLength(0); i++)
            {
                for (var j = 0; j < dCauseWeights.GetLength(1); j++)
                {
                    dCauseWeights[i, j] *= scale;
                }
            }

            for (var i = 0; i < dLabelBias.Length; i++)
            {
                dLabelBias[i] *= scale;
            }

            for (var i = 0; i < dCauseBias.Length; i++)
            {
                dCauseBias[i] *= scale;
            }

            foreach (var grad in embeddingGrads.Values)
            {
                for (var i = 0; i < grad.Length; i++)
                {
                    grad[i] *= scale;
                }
            }
        }

        private static double MaxAbs(double[] values)
        {
            var max = 0.0;
            for (var i = 0; i < values.Length; i++)
            {
                var value = Math.Abs(values[i]);
                if (value > max)
                {
                    max = value;
                }
            }

            return max;
        }
    }

    private static int Main(string[] args)
    {
        var options = ParseArgs(args);

        Console.WriteLine($"ToyConceptScaffold {Version}");
        Console.WriteLine($"Seed={options.Seed} Train={options.TrainSamples} Test={options.TestSamples} Epochs={options.Epochs} EmbedDim={options.EmbedDim}");
        Console.WriteLine();

        // Build the full token catalog once, then freeze a deterministic vocabulary.
        var tokenCatalog = BuildTokenCatalog();
        var vocabulary = BuildVocabulary(tokenCatalog);

        Console.WriteLine("Training...");
        var model = TrainMixedModel(options, vocabulary, tokenCatalog, options.Seed + 11);

        if (options.Mode.Equals("train", StringComparison.OrdinalIgnoreCase))
        {
            return 0;
        }

        if (options.Mode.Equals("sanity", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine();
            RunSanityCheck(options, model, vocabulary, tokenCatalog);
            return 0;
        }

        Console.WriteLine();
        Console.WriteLine("Evaluation: accuracy vs padding noise");
        Console.WriteLine("Cell format: LabelAcc / CauseAcc");

        var header = "PadTokens | BASELINE | CE-NEO | CE-ENG";
        Console.WriteLine(header);

        // Evaluate accuracy under increasing padding noise.
        foreach (var padTokens in options.PadLevels)
        {
            var baselineAcc = Evaluate(options, PromptFormat.Baseline, model, vocabulary, tokenCatalog, padTokens, options.Seed + 101);
            var ceNeoAcc = Evaluate(options, PromptFormat.CeNeo, model, vocabulary, tokenCatalog, padTokens, options.Seed + 202);
            var ceEngAcc = Evaluate(options, PromptFormat.CeEng, model, vocabulary, tokenCatalog, padTokens, options.Seed + 303);

            Console.WriteLine($"{padTokens,-9} | {baselineAcc.LabelAcc:0.000}/{baselineAcc.CauseAcc:0.000} | {ceNeoAcc.LabelAcc:0.000}/{ceNeoAcc.CauseAcc:0.000} | {ceEngAcc.LabelAcc:0.000}/{ceEngAcc.CauseAcc:0.000}");
        }

        Console.WriteLine();
        PrintParaphraseStress(options, vocabulary, tokenCatalog, model);

        // Print a single example in all formats to visually confirm model behavior.
        if (options.PrintDemo)
        {
            Console.WriteLine();
            PrintDemo(options, vocabulary, tokenCatalog, model);
        }

        return 0;
    }

    private static ToyModel TrainMixedModel(
        Options options,
        IReadOnlyDictionary<string, int> vocabulary,
        TokenCatalog catalog,
        int seedOffset)
    {
        // One model sees a mixture of formats so CE formats compete with baseline for capacity,
        // avoiding a trivial "CE=1.0 always" result from per-format training.
        var model = new ToyModel(vocabulary.Count, options.EmbedDim, seedOffset);
        var rng = new Random(seedOffset);
        var learningRate = 0.005;
        var trainPadLevels = new[] { 0, 5, 15, 40 };

        for (var epoch = 1; epoch <= options.Epochs; epoch++)
        {
            var totalLoss = 0.0;
            for (var i = 0; i < options.TrainSamples; i++)
            {
                var format = SampleFormat(rng, baselineWeight: 0.5, ceNeoWeight: 0.25, ceEngWeight: 0.25);
                var latent = SampleLatent(rng);
                var trainPad = trainPadLevels[rng.Next(0, trainPadLevels.Length)];
                var example = GenerateExample(format, vocabulary, catalog, rng, trainPad, options.EnableParaphrase, includeRawText: false, latentOverride: latent);
                totalLoss += model.TrainStep(example.Tokens, example.Label, example.Cause, learningRate, epoch, i + 1);
            }

            var avgLoss = totalLoss / options.TrainSamples;
            var sanity = EvaluateSample(options, PromptFormat.Baseline, model, vocabulary, catalog, padTokens: 0, seedOffset: options.Seed + 777, sampleCount: 200);
            Console.WriteLine($"Epoch {epoch}/{options.Epochs} avg loss: {avgLoss:0.0000} | sanity LabelAcc@pad0={sanity.LabelAcc:0.000}");
        }

        return model;
    }

    private static (double LabelAcc, double CauseAcc) Evaluate(
        Options options,
        PromptFormat format,
        ToyModel model,
        IReadOnlyDictionary<string, int> vocabulary,
        TokenCatalog catalog,
        int padTokens,
        int seedOffset)
    {
        var latentRng = new Random(options.Seed + 500 + padTokens);
        var renderRng = new Random(seedOffset + padTokens * 31);
        var correctLabel = 0;
        var correctCause = 0;

        for (var i = 0; i < options.TestSamples; i++)
        {
            // Sample identical latent states across formats, but allow per-format rendering/paraphrase RNGs.
            var latent = SampleLatent(latentRng);
            var example = GenerateExample(format, vocabulary, catalog, renderRng, padTokens, options.EnableParaphrase, includeRawText: false, latentOverride: latent);
            var prediction = model.Predict(example.Tokens);
            if (prediction.Label == example.Label)
            {
                correctLabel++;
            }

            if (prediction.Cause == example.Cause)
            {
                correctCause++;
            }
        }

        return ((double)correctLabel / options.TestSamples, (double)correctCause / options.TestSamples);
    }

    private static (double LabelAcc, double CauseAcc) EvaluateSample(
        Options options,
        PromptFormat format,
        ToyModel model,
        IReadOnlyDictionary<string, int> vocabulary,
        TokenCatalog catalog,
        int padTokens,
        int seedOffset,
        int sampleCount)
    {
        var latentRng = new Random(options.Seed + 500 + padTokens);
        var renderRng = new Random(seedOffset + padTokens * 31);
        var correctLabel = 0;
        var correctCause = 0;

        for (var i = 0; i < sampleCount; i++)
        {
            var latent = SampleLatent(latentRng);
            var example = GenerateExample(format, vocabulary, catalog, renderRng, padTokens, options.EnableParaphrase, includeRawText: false, latentOverride: latent);
            var prediction = model.Predict(example.Tokens);
            if (prediction.Label == example.Label)
            {
                correctLabel++;
            }

            if (prediction.Cause == example.Cause)
            {
                correctCause++;
            }
        }

        return ((double)correctLabel / sampleCount, (double)correctCause / sampleCount);
    }

    private static void PrintParaphraseStress(
        Options options,
        IReadOnlyDictionary<string, int> vocabulary,
        TokenCatalog catalog,
        ToyModel model)
    {
        const int stressRuns = 5;
        const int padTokens = 15;

        Console.WriteLine($"Paraphrase/Reorder stress (PadTokens={padTokens})");

        if (!options.EnableParaphrase)
        {
            Console.WriteLine("Paraphrase disabled; stress summary skipped.");
            return;
        }

        var baselineStats = StressRuns(options, PromptFormat.Baseline, model, vocabulary, catalog, padTokens, stressRuns);
        var ceNeoStats = StressRuns(options, PromptFormat.CeNeo, model, vocabulary, catalog, padTokens, stressRuns);
        var ceEngStats = StressRuns(options, PromptFormat.CeEng, model, vocabulary, catalog, padTokens, stressRuns);

        Console.WriteLine($"BASELINE: LabelAcc={baselineStats.LabelMean:0.000}±{baselineStats.LabelStd:0.000} CauseAcc={baselineStats.CauseMean:0.000}±{baselineStats.CauseStd:0.000}");
        Console.WriteLine($"CE-NEO:   LabelAcc={ceNeoStats.LabelMean:0.000}±{ceNeoStats.LabelStd:0.000} CauseAcc={ceNeoStats.CauseMean:0.000}±{ceNeoStats.CauseStd:0.000}");
        Console.WriteLine($"CE-ENG:   LabelAcc={ceEngStats.LabelMean:0.000}±{ceEngStats.LabelStd:0.000} CauseAcc={ceEngStats.CauseMean:0.000}±{ceEngStats.CauseStd:0.000}");
    }

    private static (double LabelMean, double LabelStd, double CauseMean, double CauseStd) StressRuns(
        Options options,
        PromptFormat format,
        ToyModel model,
        IReadOnlyDictionary<string, int> vocabulary,
        TokenCatalog catalog,
        int padTokens,
        int runs)
    {
        var labelAccs = new List<double>();
        var causeAccs = new List<double>();

        for (var i = 0; i < runs; i++)
        {
            var acc = Evaluate(options, format, model, vocabulary, catalog, padTokens, options.Seed + 900 + i * 17);
            labelAccs.Add(acc.LabelAcc);
            causeAccs.Add(acc.CauseAcc);
        }

        return (Mean(labelAccs), StdDev(labelAccs), Mean(causeAccs), StdDev(causeAccs));
    }

    private static void PrintDemo(
        Options options,
        IReadOnlyDictionary<string, int> vocabulary,
        TokenCatalog catalog,
        ToyModel model)
    {
        // Use a fixed latent state so the demo is deterministic and easy to compare across formats.
        var latent = new LatentState(1, 1, 0, 0);
        var target = ComputeTarget(latent);

        Console.WriteLine("Demo example");
        Console.WriteLine($"LATENT: slip_onset={latent.SlipOnset} shear_low={latent.ShearLow} friction_drop={latent.FrictionDrop} compliance_high={latent.ComplianceHigh}");
        Console.WriteLine($"TARGET: Label={target.Label} Cause={target.Cause}");

        var rng = new Random(options.Seed + 1234);
        PrintDemoForFormat("BASELINE", PromptFormat.Baseline, model, vocabulary, catalog, latent, rng, options.EnableParaphrase);
        PrintDemoForFormat("CE-NEO", PromptFormat.CeNeo, model, vocabulary, catalog, latent, rng, options.EnableParaphrase);
        PrintDemoForFormat("CE-ENG", PromptFormat.CeEng, model, vocabulary, catalog, latent, rng, options.EnableParaphrase);
    }

    private static void PrintDemoForFormat(
        string name,
        PromptFormat format,
        ToyModel model,
        IReadOnlyDictionary<string, int> vocabulary,
        TokenCatalog catalog,
        LatentState latent,
        Random rng,
        bool enableParaphrase)
    {
        var example = GenerateExample(format, vocabulary, catalog, rng, 0, enableParaphrase, includeRawText: true, latentOverride: latent);
        var prediction = model.Predict(example.Tokens);
        Console.WriteLine($"{name} TOKENS: {example.RawText}");
        Console.WriteLine($"PRED: Label={prediction.Label} Cause={prediction.Cause}");
    }

    private static Example GenerateExample(
        PromptFormat format,
        IReadOnlyDictionary<string, int> vocabulary,
        TokenCatalog catalog,
        Random rng,
        int padTokens,
        bool enableParaphrase,
        bool includeRawText,
        LatentState? latentOverride = null)
    {
        var latent = latentOverride ?? SampleLatent(rng);

        var target = ComputeTarget(latent);

        var tokens = format switch
        {
            PromptFormat.Baseline => RenderBaseline(latent, catalog, rng, enableParaphrase),
            PromptFormat.CeNeo => RenderScaffold(latent, catalog, rng, enableParaphrase, useNeo: true),
            PromptFormat.CeEng => RenderScaffold(latent, catalog, rng, enableParaphrase, useNeo: false),
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
        };

        AppendNoiseTokens(tokens, catalog, rng, padTokens);

        var tokenIds = tokens.Select(token => vocabulary[token]).ToArray();
        var rawText = includeRawText ? string.Join(' ', tokens) : string.Empty;

        return new Example(tokenIds, target.Label, target.Cause, latent, rawText);
    }

    private static (Label Label, Cause Cause) ComputeTarget(LatentState latent)
    {
        if (latent.SlipOnset == 1 && latent.ShearLow == 1)
        {
            return (Label.Slip, Cause.ShearLow);
        }

        if (latent.ComplianceHigh == 1 && latent.FrictionDrop == 1)
        {
            return (Label.Squish, Cause.FrictionDrop);
        }

        return (Label.Ok, Cause.None);
    }

    private static LatentState SampleLatent(Random rng)
    {
        return new LatentState(
            rng.Next(0, 2),
            rng.Next(0, 2),
            rng.Next(0, 2),
            rng.Next(0, 2));
    }

    private static List<string> RenderBaseline(LatentState latent, TokenCatalog catalog, Random rng, bool enableParaphrase)
    {
        var clauses = new List<List<string>>
        {
            PickClause(latent.SlipOnset, catalog.SlipOnsetPositive, catalog.SlipOnsetNegative, rng, enableParaphrase),
            PickClause(latent.ShearLow, catalog.ShearLowPositive, catalog.ShearLowNegative, rng, enableParaphrase),
            PickClause(latent.FrictionDrop, catalog.FrictionDropPositive, catalog.FrictionDropNegative, rng, enableParaphrase),
            PickClause(latent.ComplianceHigh, catalog.ComplianceHighPositive, catalog.ComplianceHighNegative, rng, enableParaphrase)
        };

        if (enableParaphrase)
        {
            Shuffle(clauses, rng);
        }

        var tokens = new List<string> { "observed" };
        for (var i = 0; i < clauses.Count; i++)
        {
            tokens.AddRange(clauses[i]);
            if (i < clauses.Count - 1)
            {
                tokens.Add(catalog.FillerTokens[rng.Next(0, catalog.FillerTokens.Count)]);
            }
        }

        return tokens;
    }

    private static List<string> RenderScaffold(
        LatentState latent,
        TokenCatalog catalog,
        Random rng,
        bool enableParaphrase,
        bool useNeo)
    {
        var tokens = new List<string>
        {
            useNeo ? $"slivox={latent.SlipOnset}" : $"slip_onset={latent.SlipOnset}",
            useNeo ? $"shearvon={latent.ShearLow}" : $"shear_low={latent.ShearLow}",
            useNeo ? $"frictal={latent.FrictionDrop}" : $"friction_drop={latent.FrictionDrop}",
            useNeo ? $"compliq={latent.ComplianceHigh}" : $"compliance_high={latent.ComplianceHigh}"
        };

        var distractorKeys = useNeo ? catalog.CeNeoDistractorKeys : catalog.CeEngDistractorKeys;
        foreach (var distractorKey in distractorKeys)
        {
            tokens.Add($"{distractorKey}={rng.Next(0, 2)}");
        }

        if (enableParaphrase)
        {
            Shuffle(tokens, rng);
        }

        return tokens;
    }

    private static List<string> PickClause(
        int value,
        List<List<string>> positiveClauses,
        List<List<string>> negativeClauses,
        Random rng,
        bool enableParaphrase)
    {
        var clauses = value == 1 ? positiveClauses : negativeClauses;
        if (!enableParaphrase)
        {
            return new List<string>(clauses[0]);
        }

        var idx = rng.Next(0, clauses.Count);
        return new List<string>(clauses[idx]);
    }

    private static TokenCatalog BuildTokenCatalog()
    {
        return new TokenCatalog
        {
            SlipOnsetPositive = new List<List<string>>
            {
                new() { "begins", "slips" },
                new() { "starts", "slides" },
                new() { "drifts" }
            },
            SlipOnsetNegative = new List<List<string>>
            {
                new() { "stable", "grip" },
                new() { "no", "slip" },
                new() { "holds" }
            },
            ShearLowPositive = new List<List<string>>
            {
                new() { "shear", "lateral", "wobbly" },
                new() { "shear", "low" },
                new() { "sideforce", "weak" }
            },
            ShearLowNegative = new List<List<string>>
            {
                new() { "shear", "steady" },
                new() { "lateral", "firm" },
                new() { "sideforce", "strong" }
            },
            FrictionDropPositive = new List<List<string>>
            {
                new() { "friction", "drops", "slick" },
                new() { "surface", "slickens" },
                new() { "traction", "falls" }
            },
            FrictionDropNegative = new List<List<string>>
            {
                new() { "friction", "steady" },
                new() { "surface", "dry" },
                new() { "traction", "holds" }
            },
            ComplianceHighPositive = new List<List<string>>
            {
                new() { "soft", "yields", "compliance" },
                new() { "squishy", "gives" },
                new() { "material", "softens" }
            },
            ComplianceHighNegative = new List<List<string>>
            {
                new() { "stiff", "resists" },
                new() { "rigid", "holds" },
                new() { "material", "firm" }
            },
            FillerTokens = new List<string>
            {
                "evidence",
                "because",
                "therefore",
                "and",
                "then",
                "noted",
                "signal"
            },
            CeNeoDistractorKeys = Enumerable.Range(0, 6).Select(i => $"distrax{i}").ToList(),
            CeEngDistractorKeys = Enumerable.Range(0, 6).Select(i => $"distractor{i}").ToList(),
            NoiseTokens = Enumerable.Range(0, 200).Select(i => $"noise{i}").ToList()
        };
    }

    private static Dictionary<string, int> BuildVocabulary(TokenCatalog catalog)
    {
        var tokens = new HashSet<string>();
        void AddTokens(IEnumerable<string> toAdd)
        {
            foreach (var token in toAdd)
            {
                tokens.Add(token);
            }
        }

        foreach (var clause in catalog.SlipOnsetPositive) AddTokens(clause);
        foreach (var clause in catalog.SlipOnsetNegative) AddTokens(clause);
        foreach (var clause in catalog.ShearLowPositive) AddTokens(clause);
        foreach (var clause in catalog.ShearLowNegative) AddTokens(clause);
        foreach (var clause in catalog.FrictionDropPositive) AddTokens(clause);
        foreach (var clause in catalog.FrictionDropNegative) AddTokens(clause);
        foreach (var clause in catalog.ComplianceHighPositive) AddTokens(clause);
        foreach (var clause in catalog.ComplianceHighNegative) AddTokens(clause);

        AddTokens(catalog.FillerTokens);
        AddTokens(catalog.CeNeoDistractorKeys);
        AddTokens(catalog.CeEngDistractorKeys);
        AddTokens(catalog.NoiseTokens);
        AddTokens(new[] { "observed" });

        foreach (var value in new[] { "0", "1" })
        {
            tokens.Add($"slivox={value}");
            tokens.Add($"shearvon={value}");
            tokens.Add($"frictal={value}");
            tokens.Add($"compliq={value}");

            tokens.Add($"slip_onset={value}");
            tokens.Add($"shear_low={value}");
            tokens.Add($"friction_drop={value}");
            tokens.Add($"compliance_high={value}");
        }

        foreach (var value in new[] { "0", "1" })
        {
            foreach (var key in catalog.CeNeoDistractorKeys)
            {
                tokens.Add($"{key}={value}");
            }

            foreach (var key in catalog.CeEngDistractorKeys)
            {
                tokens.Add($"{key}={value}");
            }
        }

        return tokens
            .OrderBy(token => token, StringComparer.Ordinal)
            .Select((token, index) => (token, index))
            .ToDictionary(pair => pair.token, pair => pair.index, StringComparer.Ordinal);
    }

    private static Options ParseArgs(string[] args)
    {
        var seed = 123;
        var trainSamples = 30000;
        var testSamples = 5000;
        var epochs = 8;
        var embedDim = 32;
        var padLevels = new List<int> { 0, 5, 15, 40, 80 };
        var enableParaphrase = true;
        var printDemo = false;
        var mode = "eval";

        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            switch (arg)
            {
                case "--seed":
                    seed = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--train-samples":
                    trainSamples = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--test-samples":
                    testSamples = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--epochs":
                    epochs = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--embed-dim":
                    embedDim = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--pad-levels":
                    padLevels = args[++i]
                        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                        .Select(value => int.Parse(value, CultureInfo.InvariantCulture))
                        .ToList();
                    break;
                case "--no-paraphrase":
                    enableParaphrase = false;
                    break;
                case "--print-demo":
                    printDemo = true;
                    break;
                case "--mode":
                    mode = args[++i];
                    break;
            }
        }

        return new Options(seed, trainSamples, testSamples, epochs, embedDim, padLevels, enableParaphrase, printDemo, mode);
    }

    private static string FormatName(PromptFormat format) => format switch
    {
        PromptFormat.Baseline => "BASELINE",
        PromptFormat.CeNeo => "CE-NEO",
        PromptFormat.CeEng => "CE-ENG",
        _ => format.ToString().ToUpperInvariant()
    };

    private static int ArgMax(double[] values)
    {
        var bestIndex = 0;
        var bestValue = double.NegativeInfinity;
        for (var i = 0; i < values.Length; i++)
        {
            var value = values[i];
            if (double.IsNaN(value))
            {
                continue;
            }

            if (value > bestValue)
            {
                bestValue = value;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    private static double[] Softmax(double[] logits)
    {
        var max = logits.Max();
        var exp = new double[logits.Length];
        var sum = 0.0;
        for (var i = 0; i < logits.Length; i++)
        {
            exp[i] = Math.Exp(logits[i] - max);
            sum += exp[i];
        }

        if (sum <= 0.0 || double.IsNaN(sum) || double.IsInfinity(sum))
        {
            var uniform = 1.0 / logits.Length;
            for (var i = 0; i < exp.Length; i++)
            {
                exp[i] = uniform;
            }

            return exp;
        }

        for (var i = 0; i < exp.Length; i++)
        {
            exp[i] /= sum;
        }

        return exp;
    }

    private static void Shuffle<T>(IList<T> list, Random rng)
    {
        for (var i = list.Count - 1; i > 0; i--)
        {
            var j = rng.Next(0, i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static PromptFormat SampleFormat(Random rng, double baselineWeight, double ceNeoWeight, double ceEngWeight)
    {
        var total = baselineWeight + ceNeoWeight + ceEngWeight;
        var sample = rng.NextDouble() * total;
        if (sample < baselineWeight)
        {
            return PromptFormat.Baseline;
        }

        if (sample < baselineWeight + ceNeoWeight)
        {
            return PromptFormat.CeNeo;
        }

        return PromptFormat.CeEng;
    }

    private static void AppendNoiseTokens(List<string> tokens, TokenCatalog catalog, Random rng, int padTokens)
    {
        if (padTokens <= 0)
        {
            return;
        }

        for (var i = 0; i < padTokens; i++)
        {
            tokens.Add(catalog.NoiseTokens[rng.Next(0, catalog.NoiseTokens.Count)]);
        }
    }

    private static void RunSanityCheck(
        Options options,
        ToyModel model,
        IReadOnlyDictionary<string, int> vocabulary,
        TokenCatalog catalog)
    {
        Console.WriteLine("Sanity check: token counts + noise tails + pooled norm");
        var rng = new Random(options.Seed + 707);
        var latent = SampleLatent(rng);
        var pads = new[] { 0, 80 };

        foreach (var format in new[] { PromptFormat.Baseline, PromptFormat.CeNeo, PromptFormat.CeEng })
        {
            foreach (var pad in pads)
            {
                var example = GenerateExample(format, vocabulary, catalog, rng, pad, options.EnableParaphrase, includeRawText: true, latentOverride: latent);
                var forward = model.Forward(example.Tokens);
                var tokens = example.RawText.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                var tail = tokens.Length <= 10 ? tokens : tokens[^10..];
                var norm = Math.Sqrt(forward.Pooled.Sum(value => value * value));
                Console.WriteLine($"{FormatName(format)} pad={pad} tokenCount={tokens.Length} pooledNorm={norm:0.0000}");
                Console.WriteLine($"tail: {string.Join(' ', tail)}");
            }
        }
    }

    private static double Mean(IReadOnlyList<double> values)
    {
        if (values.Count == 0)
        {
            return 0.0;
        }

        return values.Sum() / values.Count;
    }

    private static double StdDev(IReadOnlyList<double> values)
    {
        if (values.Count == 0)
        {
            return 0.0;
        }

        var mean = Mean(values);
        var variance = values.Sum(value => Math.Pow(value - mean, 2)) / values.Count;
        return Math.Sqrt(variance);
    }

    private sealed class TokenCatalog
    {
        public List<List<string>> SlipOnsetPositive { get; init; } = new();
        public List<List<string>> SlipOnsetNegative { get; init; } = new();
        public List<List<string>> ShearLowPositive { get; init; } = new();
        public List<List<string>> ShearLowNegative { get; init; } = new();
        public List<List<string>> FrictionDropPositive { get; init; } = new();
        public List<List<string>> FrictionDropNegative { get; init; } = new();
        public List<List<string>> ComplianceHighPositive { get; init; } = new();
        public List<List<string>> ComplianceHighNegative { get; init; } = new();
        public List<string> FillerTokens { get; init; } = new();
        public List<string> CeNeoDistractorKeys { get; init; } = new();
        public List<string> CeEngDistractorKeys { get; init; } = new();
        public List<string> NoiseTokens { get; init; } = new();
    }
}
