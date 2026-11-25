-- Tree optimizer benchmark (not a test: the name doesn't end in _spec, so the normal busted run skips it).
--
-- Loads each build in src/Builds/optimizer test builds/, scores its current (hand-made) tree with the optimizer's own
-- scoring, then runs the optimizer with the same point budget starting from the current tree and from scratch, and
-- reports an anytime curve (best score so far at fixed CPU times), evaluation counts and a time profile.
--
-- Run from the repo root (Git Bash). Output dir is optional; JSON/CSV is written there when it is mounted:
--   MSYS_NO_PATHCONV=1 docker run --rm -e HOME=/tmp -v "$(pwd -W)":/workdir:ro -w /workdir \
--     -v "<writable dir>":/out -e BENCH_OUT=/out -e BENCH_BUILDS=RF -e BENCH_TIME=60 \
--     ghcr.io/pathofbuildingcommunity/pathofbuilding-tests:latest busted --lua=luajit ../spec/bench/TreeOptimizerBench.lua
--
-- Environment variables (all optional):
--   BENCH_BUILDS   comma separated substrings; only builds whose file name contains one of them run (default: all)
--   BENCH_DIR      directory of build XMLs, relative to src (default: "Builds/optimizer test builds")
--   BENCH_MODES    comma separated: "current", "scratch" (default: both)
--   BENCH_TIME     optimizer time limit in CPU seconds (default: 60)
--   BENCH_SEED     optimizer random seed (default: 1)
--   BENCH_PROFILE  "0" disables method timing (default: on; it costs a few percent of run time)
--   BENCH_OUT      writable directory for bench_<tag>.json / bench_<tag>.csv
--   BENCH_TAG      suffix for the output files (default: "run")
--   BENCH_ALLOW_MISSING_LUT  "1" runs builds even when timeless jewel data (src/Data/TimelessJewelData/*.bin) is missing;
--                  their scores are NOT comparable with runs that have it (default: such builds fail with an error)
--   BENCH_OPTIONS  extra optimizer options as Lua table fields, e.g. "useModel=false, maxPathLength=5"
--
--   BENCH_EVALS    N: run on the optimizer's evaluation clock (clock = "evals") with a budget of N calculations (timeLimit
--                  = N * BENCH_EVALTIME). Runs are then deterministic for a seed whatever else the machine is doing, and
--                  checkpoints are reported by evaluation count (250, 500, 1000, ... up to N) instead of seconds.
--   BENCH_CLOCK    "evals" is the same as giving BENCH_EVALS (default budget 4000); "time" (default) is the CPU clock
--   BENCH_EVALTIME nominal seconds per calculation under the evals clock (default 0.015)
--   BENCH_PROBE    path of a Lua file returning function(optimizer, result), run on the finished optimizer (diagnostics)
--   BENCH_DUMPLOG  any value: the JSON also carries every prediction log entry (per discovered field predicted and real values with the structured model)
--
-- Every run logs the model's predictions against the calculations that check them (optimizer.predictionLog) and
-- reports per build: the count, mean absolute error of the log score, Spearman rank correlation, the share of predicted
-- improvements that really improved, and the error by number of nodes changed from the model's base tree.
--
-- Times are os.clock() CPU seconds, the same clock the optimizer uses when running headless (time mode).
--
-- Builds with a Glorious Vanity jewel need src/Data/TimelessJewelData/GloriousVanity.bin (written by the desktop app the
-- first time it loads such a build); headless runs can't decompress the .zip parts.
--
-- Running builds in parallel containers (one per BENCH_BUILDS filter) is fine: os.clock is per process CPU time.

local dkjson = require("dkjson")

local TIME_CHECKPOINTS = { 1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300 }
local EVAL_CHECKPOINTS = { 250, 500, 1000, 2000, 4000, 8000, 16000 }
local CHECKPOINTS = TIME_CHECKPOINTS -- the list in use; switched to EVAL_CHECKPOINTS in evals mode
local evalMode = false

local function env(name, default)
	local value = os.getenv(name)
	if value == nil or value == "" then
		return default
	end
	return value
end

local function split(str)
	local list = { }
	for part in (str or ""):gmatch("[^,]+") do
		table.insert(list, (part:gsub("^%s+", ""):gsub("%s+$", "")))
	end
	return list
end

local function out(fmt, ...)
	io.stdout:write(string.format(fmt, ...))
	io.stdout:flush()
end

-- Methods timed by the profiler. Self time excludes time spent in other timed methods, so the self times add up to the
-- time inside Run (plus untimed glue, reported as "other").
local PROFILED = { "calcFunc", "Evaluate", "Track", "Frontier", "Reachable", "RemovableBranches", "CopyState", "Prune",
	"Fill", "Refine", "Reroute", "FarSwap", "Perturb", "Trim", "ConnectMustHaves", "BestMasteryEffect", "BuildModel",
	"ModelRound", "ModelEvaluate", "ModelFields", "IsEligible" }
-- Methods whose result says whether they found an improving tree
local SUCCESS_METHODS = { Reroute = true, FarSwap = true, ModelRound = true }

local function instrument(optimizer, profile)
	local stats = {
		events = { },        -- every time the best tree improves: { t, score, value, deficit, evals, phase }
		checkpoints = { },   -- [i] = { evals } when CPU time first crosses CHECKPOINTS[i]
		phaseTime = { },     -- CPU seconds while optimizer.phase had each value (sampled at timed method boundaries)
		phaseEvals = { },    -- full calculations per phase
		phaseImprovements = { }, -- best-tree improvements per phase: { count, gain }
		self = { }, total = { }, calls = { }, success = { },
		calcTime = 0,
	}
	local clock = os.clock
	local stack = { }        -- child time accumulators
	local lastTick, lastPhase = clock(), optimizer.phase
	local nextCheckpoint = 1

	local function tick()
		local t = clock()
		stats.phaseTime[lastPhase] = (stats.phaseTime[lastPhase] or 0) + (t - lastTick)
		lastTick, lastPhase = t, optimizer.phase
		local start = optimizer.startTime
		if start and not evalMode then
			while CHECKPOINTS[nextCheckpoint] and t - start >= CHECKPOINTS[nextCheckpoint] do
				stats.checkpoints[nextCheckpoint] = { evals = optimizer.evalCount }
				nextCheckpoint = nextCheckpoint + 1
			end
		end
		return t
	end
	stats.tick = tick

	local function wrap(name, fn, isMethod)
		stats.calls[name], stats.self[name], stats.total[name] = 0, 0, 0
		return function(...)
			stats.calls[name] = stats.calls[name] + 1
			local t0 = tick()
			local depth = #stack + 1
			stack[depth] = 0
			local results = { fn(...) }
			local t1 = tick()
			local elapsed = t1 - t0
			local child = stack[depth]
			stack[depth] = nil
			if depth > 1 then
				stack[depth - 1] = stack[depth - 1] + elapsed
			end
			stats.total[name] = stats.total[name] + elapsed
			stats.self[name] = stats.self[name] + elapsed - child
			if SUCCESS_METHODS[name] and results[1] ~= nil then
				stats.success[name] = (stats.success[name] or 0) + 1
			end
			return unpack(results, 1, table.maxn(results))
		end
	end

	-- Anytime curve: Track is the only place the best tree changes
	local track = optimizer.Track
	optimizer.Track = function(self, state)
		local before = self.best
		track(self, state)
		if self.best ~= before then
			local eval = self.best.eval
			local t = os.clock() - (self.startTime or os.clock())
			local prev = stats.events[#stats.events]
			table.insert(stats.events, { t = t, score = eval.score, value = eval.value, deficit = eval.deficit, evals = self.evalCount, phase = self.phase, count = self.best.count })
			local phaseStat = stats.phaseImprovements[self.phase] or { count = 0, gain = 0 }
			phaseStat.count = phaseStat.count + 1
			phaseStat.gain = phaseStat.gain + (prev and (eval.score - prev.score) or 0)
			stats.phaseImprovements[self.phase] = phaseStat
		end
	end

	-- Count full calculations per phase and time inside the calculator itself
	local calcFunc = optimizer.calcFunc
	optimizer.calcFunc = function(...)
		stats.phaseEvals[optimizer.phase] = (stats.phaseEvals[optimizer.phase] or 0) + 1
		local t0 = os.clock()
		local a, b, c = calcFunc(...)
		stats.calcTime = stats.calcTime + os.clock() - t0
		return a, b, c
	end

	if profile then
		for _, name in ipairs(PROFILED) do
			if name == "calcFunc" then
				optimizer.calcFunc = wrap(name, optimizer.calcFunc)
			elseif name == "Track" then
				optimizer.Track = wrap(name, optimizer.Track)
			else
				optimizer[name] = wrap(name, optimizer[name])
			end
		end
	else
		-- Still sample checkpoints on every calculation
		local inner = optimizer.calcFunc
		optimizer.calcFunc = function(...)
			tick()
			return inner(...)
		end
	end
	return stats
end

-- Headless Inflate is a stub, so timeless jewel data (e.g. Glorious Vanity) can't be decompressed and builds using it
-- fail to calculate. Point the loader at the uncompressed .bin cache the desktop app writes next to the .zip parts
-- (src/Data/TimelessJewelData/*.bin, gitignored) when one exists.
local missingLUT = 0
local function useUncompressedJewelData()
	-- Without the data a timeless jewel silently does nothing, which changes the build's stats (and the hand-made
	-- tree's score by 10-15%), so count the warnings and refuse to report numbers that aren't comparable.
	local conPrintf = _G.ConPrintf
	_G.ConPrintf = function(fmt, ...)
		if type(fmt) == "string" and fmt:find("Missing LUT", 1, true) then
			missingLUT = missingLUT + 1
		end
		return conPrintf(fmt, ...)
	end
	-- busted runs spec files in their own environment, so set the global on _G
	local newFileSearch = _G.NewFileSearch
	_G.NewFileSearch = function(spec, ...)
		if type(spec) == "string" and spec:match("TimelessJewelData/[^/]+%.bin$") then
			local f = io.open(spec, "rb")
			if f then
				f:close()
				return {
					GetFileName = function() return spec:match("[^/]+$") end,
					GetFileModifiedTime = function() return math.huge end,
				}
			end
		end
		return newFileSearch(spec, ...)
	end
end

local function findBuilds(dir, filters)
	local files = { }
	for file in lfs.dir(dir) do
		if file:match("%.xml$") then
			local keep = not filters[1]
			for _, filter in ipairs(filters) do
				if file:lower():find(filter:lower(), 1, true) then
					keep = true
				end
			end
			if keep then
				table.insert(files, file)
			end
		end
	end
	table.sort(files)
	return files
end

local function readFile(path)
	local f = assert(io.open(path, "rb"))
	local text = f:read("*a")
	f:close()
	return text
end

local function loadBuild(dir, file)
	loadBuildFromXML(readFile(dir .. "/" .. file), file:gsub("%.xml$", ""))
	assert(build.spec, "build has no tree")
	assert(build.calcsTab.mainEnv, "build has no calculated output")
end

-- Options as TreeTabClass:StartOptimizer builds them with the popup's defaults (Power Metric, the points the current
-- tree uses, the tree's marks and requirements)
local function uiOptions(mode, timeLimit, seed, extra)
	local spec = build.spec
	local mustHave, mustNot = { }, { }
	for nodeId, mark in pairs(spec.optimizerMarks or { }) do
		if spec.nodes[nodeId] then
			if mark == "must" then
				mustHave[nodeId] = true
			else
				mustNot[nodeId] = true
			end
		end
	end
	local powerStat
	for _, stat in ipairs(data.powerStatList) do
		if stat.stat and not stat.ignoreForNodes then
			powerStat = stat
			break
		end
	end
	local options = {
		mustHave = mustHave,
		mustNot = mustNot,
		points = nil,
		requirements = copyTable(spec.optimizerRequirements or { }),
		fromScratch = mode == "scratch",
		powerStat = powerStat,
		timeLimit = timeLimit,
		seed = seed,
	}
	for k, v in pairs(extra) do
		options[k] = v
	end
	return options
end

local function bestAt(events, t)
	local best
	for _, e in ipairs(events) do
		if e[evalMode and "evals" or "t"] <= t then
			best = e
		end
	end
	return best
end

local function round(x, digits)
	if type(x) ~= "number" then
		return x
	end
	local m = 10 ^ (digits or 4)
	return math.floor(x * m + 0.5) / m
end

-- Average ranks (ties share the mean rank)
local function ranks(values)
	local order = { }
	for i = 1, #values do
		order[i] = i
	end
	table.sort(order, function(a, b) if values[a] ~= values[b] then return values[a] < values[b] end return a < b end)
	local result = { }
	local i = 1
	while i <= #order do
		local j = i
		while j < #order and values[order[j + 1]] == values[order[i]] do
			j = j + 1
		end
		for k = i, j do
			result[order[k]] = (i + j) / 2
		end
		i = j + 1
	end
	return result
end

local function pearson(x, y)
	local n = #x
	if n < 3 then
		return nil
	end
	local mx, my = 0, 0
	for i = 1, n do
		mx, my = mx + x[i], my + y[i]
	end
	mx, my = mx / n, my / n
	local sxy, sxx, syy = 0, 0, 0
	for i = 1, n do
		sxy = sxy + (x[i] - mx) * (y[i] - my)
		sxx = sxx + (x[i] - mx) ^ 2
		syy = syy + (y[i] - my) ^ 2
	end
	if sxx == 0 or syy == 0 then
		return nil
	end
	return sxy / math.sqrt(sxx * syy)
end

local BUCKETS = { { 1, 5 }, { 6, 10 }, { 11, 20 }, { 21, math.huge } }

-- Model predictions against the calculations that checked them (optimizer.predictionLog)
local function predictionStats(log)
	local stats = { count = #log }
	if #log == 0 then
		return stats
	end
	local predicted, real, absErr, signed = { }, { }, 0, 0
	local predBetter, bothBetter, realBetterCount, defErr, defCount = 0, 0, 0, 0, 0
	local buckets = { }
	for i, b in ipairs(BUCKETS) do
		buckets[i] = { from = b[1], to = b[2], count = 0, absErr = 0, signed = 0, predBetter = 0, bothBetter = 0 }
	end
	for i, p in ipairs(log) do
		predicted[i], real[i] = p.predScore, p.realScore
		local err = p.predScore - p.realScore
		absErr, signed = absErr + math.abs(err), signed + err
		defErr, defCount = defErr + math.abs(p.predDeficit - p.realDeficit), defCount + 1
		if p.predBetter then
			predBetter = predBetter + 1
			if p.realBetter then
				bothBetter = bothBetter + 1
			end
		end
		if p.realBetter then
			realBetterCount = realBetterCount + 1
		end
		for _, bucket in ipairs(buckets) do
			if p.changed >= bucket.from and p.changed <= bucket.to then
				bucket.count = bucket.count + 1
				bucket.absErr = bucket.absErr + math.abs(err)
				bucket.signed = bucket.signed + err
				if p.predBetter then
					bucket.predBetter = bucket.predBetter + 1
					if p.realBetter then
						bucket.bothBetter = bucket.bothBetter + 1
					end
				end
			end
		end
	end
	stats.maeScore = absErr / #log
	stats.bias = signed / #log -- predicted minus real: positive means the model is optimistic
	stats.maeDeficit = defErr / defCount
	stats.spearman = pearson(ranks(predicted), ranks(real))
	stats.predBetter = predBetter
	stats.precision = predBetter > 0 and bothBetter / predBetter or nil
	stats.realBetter = realBetterCount
	for _, bucket in ipairs(buckets) do
		if bucket.count > 0 then
			bucket.maeScore = bucket.absErr / bucket.count
			bucket.bias = bucket.signed / bucket.count
			bucket.precision = bucket.predBetter > 0 and bucket.bothBetter / bucket.predBetter or nil
		end
	end
	stats.buckets = buckets
	return stats
end

-- Cheap fingerprint of a run's improvement events, to compare runs exactly
local function fingerprint(events, final)
	local h = 5381
	local text = { }
	for _, e in ipairs(events) do
		text[#text + 1] = string.format("%d:%.17g:%.17g", e.evals, e.score, e.deficit)
	end
	text[#text + 1] = string.format("%.17g", final)
	local s = table.concat(text, "|")
	for i = 1, #s do
		h = (h * 33 + s:byte(i)) % 4294967296
	end
	return string.format("%08x/%d", h, #events)
end

local function runOne(dir, file, mode, settings)
	missingLUT = 0
	loadBuild(dir, file)
	if missingLUT > 0 and env("BENCH_ALLOW_MISSING_LUT", "0") ~= "1" then
		error("timeless jewel data is missing (" .. missingLUT .. " 'Missing LUT' warnings): the jewel would do nothing and scores would differ from runs with " ..
			"src/Data/TimelessJewelData/*.bin. Copy the .bin files (gitignored, written by the desktop app) into this checkout, or set BENCH_ALLOW_MISSING_LUT=1", 0)
	end
	collectgarbage("collect")
	local TreeOptimizer = LoadModule("Modules/TreeOptimizer")
	local optimizer = TreeOptimizer.new(build, uiOptions(mode, settings.timeLimit, settings.seed, settings.extra))
	local stats = instrument(optimizer, settings.profile)
	local cpu0 = os.clock()
	local ok, result = pcall(optimizer.Run, optimizer)
	local runTime = os.clock() - cpu0
	stats.tick()
	if not ok then
		return { build = file, mode = mode, error = tostring(result) }
	end
	if result.error then
		return { build = file, mode = mode, error = result.error }
	end
	local probe = env("BENCH_PROBE", nil)
	if probe then
		-- Diagnostics hook: a Lua file returning function(optimizer, result) run on the finished optimizer
		optimizer.stopped, optimizer.timeUp, optimizer.deadline, optimizer.maxEvals = false, false, nil, math.huge
		local okp, errp = pcall(assert(loadfile(probe))(), optimizer, result)
		if not okp then out("probe error: %s", tostring(errp)) end
	end
	local init = result.initialEval
	local function rel(e)
		if not e then
			return nil
		end
		return { ratio = e.value / init.value, dlog = e.score - init.score, deficit = e.deficit, evals = e.evals, t = e.t }
	end
	local curve = { }
	for i, t in ipairs(CHECKPOINTS) do
		if t <= settings.limit + 1e-6 or i == 1 then
			local e = bestAt(stats.events, t)
			local entry = rel(e) or { }
			entry.at = t
			entry.evalsAt = evalMode and t or (stats.checkpoints[i] and stats.checkpoints[i].evals)
			table.insert(curve, entry)
		end
	end
	local finalEval = result.bestEval
	local afterFill = result.afterFillEval
	local profile
	if settings.profile then
		profile = { }
		for _, name in ipairs(PROFILED) do
			if stats.calls[name] and stats.calls[name] > 0 then
				table.insert(profile, { name = name, calls = stats.calls[name], self = stats.self[name], total = stats.total[name], success = stats.success[name] })
			end
		end
		table.sort(profile, function(a, b) return a.self > b.self end)
	end
	local phases = { }
	for phase, time in pairs(stats.phaseTime) do
		local imp = stats.phaseImprovements[phase] or { }
		table.insert(phases, { phase = phase, time = time, evals = stats.phaseEvals[phase] or 0, improvements = imp.count or 0, gain = imp.gain or 0 })
	end
	table.sort(phases, function(a, b) return a.time > b.time end)
	local violations = finalEval.violations or { }
	-- Notables, keystones and masteries the final tree gained and lost against the hand-made tree
	local added, removed = optimizer:Diff(result.best)
	local function names(list)
		local out = { }
		for _, entry in ipairs(list) do
			if entry.node.type ~= "Normal" then
				table.insert(out, (entry.node.name or "?") .. (entry.effectId and "*" or ""))
			end
		end
		table.sort(out)
		return out
	end
	return {
		gained = names(added),
		lost = names(removed),
		build = file,
		mode = mode,
		points = optimizer.points,
		mustHave = (function() local n = 0 for _ in pairs(optimizer.mustHave) do n = n + 1 end return n end)(),
		initial = { value = init.value, score = init.score, deficit = init.deficit, violations = init.violations },
		afterFill = { ratio = afterFill.value / init.value, dlog = afterFill.score - init.score, deficit = afterFill.deficit },
		final = { ratio = finalEval.value / init.value, dlog = finalEval.score - init.score, deficit = finalEval.deficit, value = finalEval.value, count = result.best.count, violations = violations },
		fillTime = result.fillTime,
		fillEvals = result.fillEvals,
		thoroughTime = result.thoroughTime,
		thoroughEvals = result.thoroughEvals,
		time = result.time,
		runTime = runTime,
		evals = result.evals,
		cacheHits = result.cacheHits,
		calcTime = stats.calcTime,
		msPerEval = result.evals > 0 and stats.calcTime / result.evals * 1000 or 0,
		modelRounds = result.modelRounds,
		localOptima = result.localOptima,
		curve = curve,
		accuracy = predictionStats(optimizer.predictionLog or { }),
		structured = optimizer.structured,
		discovered = optimizer.discPaths,
		predictionLog = env("BENCH_DUMPLOG", nil) and optimizer.predictionLog or nil,
		fingerprint = fingerprint(stats.events, finalEval.score),
		events = stats.events,
		phases = phases,
		profile = profile,
	}
end

local function fmtRatio(entry)
	if not entry or not entry.ratio then
		return "    -   "
	end
	return string.format("%7.4f%s", entry.ratio, (entry.deficit or 0) > 0 and "!" or " ")
end

local function printResult(r)
	if r.error then
		out("\n### %s [%s] ERROR: %s\n", r.build, r.mode, r.error)
		return
	end
	out("\n### %s [%s]  points=%d mustHave=%d\n", r.build, r.mode, r.points, r.mustHave)
	out("hand-made: value=%.6g score=%.5f deficit=%.3g %s\n", r.initial.value, r.initial.score, r.initial.deficit, table.concat(r.initial.violations or { }, "; "))
	out("after fill (%.1fs, %d evals%s): ratio=%.4f dlog=%+.4f deficit=%.3g\n", r.fillTime, r.fillEvals, r.thoroughTime and string.format("; thorough fill done at %.1fs, %d evals", r.thoroughTime, r.thoroughEvals) or "", r.afterFill.ratio, r.afterFill.dlog, r.afterFill.deficit)
	out("final (%.1fs): ratio=%.4f dlog=%+.4f deficit=%.3g nodes=%d %s\n", r.time, r.final.ratio, r.final.dlog, r.final.deficit, r.final.count, table.concat(r.final.violations, "; "))
	out("evals=%d cacheHits=%d calc=%.1fs (%.1f%% of run) ms/eval=%.1f modelRounds=%d localOptima=%d\n", r.evals, r.cacheHits, r.calcTime,
		100 * r.calcTime / math.max(r.runTime, 1e-9), r.msPerEval, r.modelRounds, r.localOptima)
	out("curve:")
	for _, c in ipairs(r.curve) do
		out("  %ds=%s(%s)", c.at, c.ratio and string.format("%.4f", c.ratio) or "-", c.evalsAt or "-")
	end
	out("\n")
	out("%-14s %8s %7s %6s %9s\n", "phase", "cpu s", "evals", "impr", "gain(log)")
	for _, p in ipairs(r.phases) do
		out("%-14s %8.2f %7d %6d %9.4f\n", tostring(p.phase), p.time, p.evals, p.improvements, p.gain)
	end
	if r.profile then
		out("%-18s %8s %8s %8s %8s\n", "method", "calls", "self s", "total s", "success")
		for _, p in ipairs(r.profile) do
			out("%-18s %8d %8.2f %8.2f %8s\n", p.name, p.calls, p.self, p.total, p.success or "")
		end
	end
	local a = r.accuracy
	if a and a.count > 0 then
		out("model predictions: n=%d maeLog=%.4f bias=%+.4f maeDeficit=%.3g spearman=%s predictedBetter=%d realBetter=%d precision=%s\n", a.count, a.maeScore, a.bias, a.maeDeficit, a.spearman and string.format("%.3f", a.spearman) or "-", a.predBetter, a.realBetter, a.precision and string.format("%.2f", a.precision) or "-")
		for _, b in ipairs(a.buckets) do
			if b.count > 0 then
				out("  changed %d-%s: n=%d maeLog=%.4f bias=%+.4f precision=%s\n", b.from, b.to == math.huge and "+" or tostring(b.to), b.count, b.maeScore, b.bias, b.precision and string.format("%.2f", b.precision) or "-")
			end
		end
	else
		out("model predictions: none\n")
	end
	out("fingerprint: %s\n", r.fingerprint)
	out("improvements (t, ratio, phase):")
	local init = r.initial
	for i, e in ipairs(r.events) do
		if i <= 3 or i > #r.events - 25 then
			out(" %.1f:%.4f:%s", e.t, e.value / init.value, e.phase)
		elseif i == 4 then
			out(" ...")
		end
	end
	out("\n")
end

local function printSummary(results, timeLimit)
	out("\n==== Summary (ratio = optimizer value / hand-made value; ! = constraint deficit > 0) ====\n")
	local cols = evalMode and { 250, 500, 1000, 2000, 4000 } or { 5, 10, 30, 60 }
	out("%-36s %-7s %10s %8s", "build", "mode", "hand-made", "fill")
	for _, t in ipairs(cols) do
		out(" %8s", t .. (evalMode and "e" or "s"))
	end
	out(" %8s %7s %6s %7s %8s\n", "final", "dlog", "evals", "ms/eval", "fillTime")
	for _, r in ipairs(results) do
		if r.error then
			out("%-36s %-7s ERROR %s\n", r.build:sub(1, 36), r.mode, r.error)
		else
			out("%-36s %-7s %10.4g %s", r.build:sub(1, 36), r.mode, r.initial.value, fmtRatio(r.afterFill))
			for _, t in ipairs(cols) do
				local e = t <= timeLimit and bestAt(r.events, t)
				out(" %s", e and fmtRatio({ ratio = e.value / r.initial.value, deficit = e.deficit }) or "    -   ")
			end
			out(" %s %+7.4f %6d %7.1f %8.1f\n", fmtRatio(r.final), r.final.dlog, r.evals, r.msPerEval, r.fillTime)
		end
	end
end

local function writeOutputs(results, dir, tag)
	local path = dir .. "/bench_" .. tag .. ".json"
	local f = io.open(path, "w")
	if not f then
		out("Could not write %s\n", path)
		return
	end
	f:write(dkjson.encode(results, { indent = true }))
	f:close()
	local csv = io.open(dir .. "/bench_" .. tag .. ".csv", "w")
	csv:write("build,mode,handmade_value,handmade_deficit,fill_ratio,fill_time,t1,t2,t5,t10,t20,t30,t60,final_ratio,final_dlog,final_deficit,evals,ms_per_eval,calc_time,run_time\n")
	for _, r in ipairs(results) do
		if not r.error then
			local row = { r.build, r.mode, r.initial.value, r.initial.deficit, r.afterFill.ratio, r.fillTime }
			for _, t in ipairs(evalMode and { 250, 500, 1000, 2000, 4000 } or { 1, 2, 5, 10, 20, 30, 60 }) do
				local e = bestAt(r.events, t)
				table.insert(row, e and round(e.value / r.initial.value, 5) or "")
			end
			for _, v in ipairs({ r.final.ratio, r.final.dlog, r.final.deficit, r.evals, r.msPerEval, r.calcTime, r.runTime }) do
				table.insert(row, round(v, 5))
			end
			for i, v in ipairs(row) do
				row[i] = tostring(v):gsub(",", ";")
			end
			csv:write(table.concat(row, ",") .. "\n")
		end
	end
	csv:close()
	out("Wrote %s (+ .csv)\n", path)
end

describe("TreeOptimizer benchmark", function()
	it("runs", function()
		useUncompressedJewelData()
		local dir = env("BENCH_DIR", "Builds/optimizer test builds")
		local evalBudget = tonumber(env("BENCH_EVALS", "")) or (env("BENCH_CLOCK", "time") == "evals" and 4000 or nil)
		local evalTime = tonumber(env("BENCH_EVALTIME", "0.015"))
		local settings = {
			timeLimit = tonumber(env("BENCH_TIME", "60")),
			seed = tonumber(env("BENCH_SEED", "1")),
			profile = env("BENCH_PROFILE", "1") ~= "0",
			extra = { logPredictions = true },
		}
		evalMode = evalBudget ~= nil
		CHECKPOINTS = evalMode and EVAL_CHECKPOINTS or TIME_CHECKPOINTS
		settings.limit = settings.timeLimit
		if evalMode then
			settings.timeLimit = evalBudget * evalTime
			settings.limit = evalBudget
			settings.extra.clock = "evals"
			settings.extra.evalTime = evalTime
		end
		local extraText = env("BENCH_OPTIONS", nil)
		if extraText then
			for k, v in pairs(assert(loadstring("return { " .. extraText .. " }"))()) do settings.extra[k] = v end
		end
		local modes = split(env("BENCH_MODES", "current,scratch"))
		local files = findBuilds(dir, split(env("BENCH_BUILDS", "")))
		out("\nTree optimizer bench: %d build(s), modes=%s, budget=%s%s, seed=%d, profile=%s\n", #files, table.concat(modes, "/"), settings.limit, evalMode and " evals" or "s", settings.seed, tostring(settings.profile))
		local results = { }
		for _, file in ipairs(files) do
			for _, mode in ipairs(modes) do
				local ok, r = pcall(runOne, dir, file, mode, settings)
				if not ok then
					r = { build = file, mode = mode, error = tostring(r) }
				end
				printResult(r)
				table.insert(results, r)
			end
		end
		printSummary(results, settings.limit)
		local outDir = env("BENCH_OUT", nil)
		if outDir then
			writeOutputs(results, outDir, env("BENCH_TAG", "run"))
		end
	end)
end)
