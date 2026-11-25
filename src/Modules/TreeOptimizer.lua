-- Path of Building
--
-- Module: Tree Optimizer
-- Searches for a passive tree that maximises a power stat (Power Metric by default) for a fixed point budget,
-- subject to must-have / must-not-have nodes and attribute requirements.
--
-- The search never touches the build's spec while running: every candidate tree is evaluated through the
-- misc calculator using addNodes / removeNodes overrides relative to the current spec.
--
-- Phases:
--   1. Seed: start from the current tree (or from the class start), drop must-not nodes and anything they cut off.
--   2. Connect: add the cheapest path to every must-have node (greedy Steiner tree heuristic).
--   3. Trim: while over budget, remove the branch that costs the least score per point.
--   4. Fill: lazy greedy - repeatedly add the path with the best score gain per point. Never-evaluated paths take the
--      estimate of a node with the same kind of stats, and a mastery's effects are ranked once per kind of mastery.
--      Out-of-date estimates rank at 85% of their value (FILL_SLACK), which saves a third of the calculations.
--      From scratch, a quick fill comes first: it leaves plain travel nodes out of the targets it measures (they are
--      still part of the paths to notables and masteries), which takes a third to a half of the calculations, so a full
--      valid tree exists after a few seconds. A thorough fill from the same start follows, measuring everything again
--      (the quick one's estimates are dropped: they undervalue what it skipped), provided the quick one was cheap enough
--      for the thorough one to fit in half the time limit; the better of the two trees is what the search continues with.
--      Then, if the tree misses its requirements, Repair swaps branches for the paths that remove the most deficit per
--      point, from anywhere within 12 points. A tree that meets the requirements always beats one that doesn't.
--   5. Refine: remove a branch, refill without it, keep the result if it scored better. When no branch helps, swap a
--      travel node for a neighbour (Reroute), or take the path to a notable/keystone 5-12 points away and pay for it
--      with the cheapest branches anywhere in the tree (FarSwap).
--   6. Until time runs out (the deadline is checked after every calculation, so Fill, Repair and the model's
--      measuring stop on time too). The model ranks trees by the same deficit-first IsBetter as the real search:
--      - A best tree that still misses its requirements gets one Repair before anything else is tried on it.
--        So does a tree the model found that beats the base's stat but misses a requirement.
--      - Model rounds first: each measures one more chunk of nodes around the best tree (nearest and most useful
--        first, reusing measurements from earlier base trees as estimates), searches a cheap additive model of the
--        score with the moves above, and checks its best trees with full calculations (see "Node value model").
--        Rounds go on while they find better trees or have nodes left to measure.
--        Predictions are optimistic, and the more so the more nodes a tree changes from the model's base (the best of
--        many is picked, and sums of single-node effects drift: caps and saturating stats like total EHP). So each
--        prediction is lowered by a penalty learned from this run's own checks (mean predicted - real per distance
--        band, interpolated by changed-node count: RecordCheck, DistancePenalty), which steers both the model search
--        and the choice of trees to check. The optional structured model is likewise kept only while its checks are
--        no worse than the plain sum's (StructuredWins).
--      - When the model rests (failed rounds with nothing left to measure), Refine runs on the best tree until its
--        first improving move, so the model gets its turn on the better tree; at a local optimum, far swaps then
--        random perturbation, each followed by Refine. Perturbation alternates between removing a few branches (more
--        of them the more kicks fail) and removing a whole region around a notable (PerturbRegion). The search on the
--        model uses the same kicks.
--
local m_max = math.max
local m_min = math.min
local m_floor = math.floor
local m_log = math.log
local t_insert = table.insert
local t_sort = table.sort
local t_concat = table.concat

local TreeOptimizer = { }
TreeOptimizer.__index = TreeOptimizer

-- Paths cannot leave masteries (they're leaves), same as PassiveSpec:BuildAllDependsAndPaths
local function canLeave(node)
	return node.type ~= "Mastery"
end

-- Wall clock in seconds. GetTime is a stub returning 0 when running headless, so fall back to CPU time there.
local function now()
	local ms = GetTime and GetTime() or 0
	return ms > 0 and ms / 1000 or os.clock()
end

-- Evaluation clock (option clock = "evals"): time is counted in calculations instead of read from the machine, so a
-- run depends only on its seed, never on CPU speed or load. One calculation takes `evalTime` seconds. Work that
-- calculates nothing is charged too: each model prediction costs PREDICTION_COST of a calculation, and every read of
-- the clock a little more (CLOCK_READ_COST), which keeps any loop that only reads the clock moving.
local DEFAULT_EVAL_TIME = 0.015
-- Measured: a search of the model makes ~3000 predictions per CPU second against ~80 calculations
local PREDICTION_COST = 0.027
local CLOCK_READ_COST = 0.0001

-- Log-score cost of each missing attribute point. 10 missing points cost ~18% of the stat.
local ATTRIBUTE_PENALTY = 0.02
-- Log-score cost of each deficit point when comparing trees that don't meet the requirements (see Merit). Large enough
-- that nearly any step towards meeting them is worth more than the stat it costs.
local INFEASIBLE_PENALTY = 1
local EPSILON = 1e-9

---@class TreeOptimizerOptions
---@field mustHave table<number, boolean|number>? node id -> true, or a mastery effect id for mastery nodes
---@field mustNot table<number, boolean>? node id -> true
---@field points number? Passive point budget (excluding ascendancy). Defaults to the points used by the current tree.
---@field requirements table<string, number>? Manual requirements: attributes (Str, Dex, Int, Omni) override the calculated
---requirement, ManaUnreserved sets the minimum unreserved mana (default 0), SustainCost = 0 stops requiring recovery to
---keep up with the main skill's cost. e.g. { Str = 100, ManaUnreserved = 50 }
---@field fromScratch boolean? Start from the class start instead of the current tree.
---@field powerStat table? Entry of data.powerStatList to maximise. Defaults to the Power Metric.
---@field maxPathLength number? Furthest an added path can reach in one step.
---@field maxBranchSize number? Largest branch the refine phase may remove in one move.
---@field timeLimit number? Seconds the whole run may take (the initial fill always completes).
---@field clock string? "evals" counts time in calculations (see DEFAULT_EVAL_TIME) so runs are deterministic; default is real time.
---@field evalTime number? Seconds one calculation counts for under the evals clock (default 0.015).
---@field structuredModel boolean? Predict the stat by running the score function on a predicted output (default false: it helps some builds and hurts others, see ModelFields), instead of summing log deltas of the stat itself. Falls back to the sum when its checks are worse.
---@field scoreFunction fun(output: table): number? Replaces the power stat as the number to maximise (a function of the output table alone). Used by tests.
---@field distancePenalty boolean? Subtract from each model prediction the optimism the model's checks showed at that distance from its base (default true).
---@field logPredictions boolean? Record model predictions against the calculations that check them in self.predictionLog.
---@field maxEvals number? Hard cap on the number of calculations.
---@field onProgress fun(info: table)? Called after every calculation. Running inside a coroutine, it may yield.
---@field shouldStop fun(): boolean? Polled after every calculation; returning true ends the run with the best tree so far.

function TreeOptimizer.new(build, options)
	local self = setmetatable({ }, TreeOptimizer)
	options = options or { }
	self.build = build
	self.spec = build.spec
	self.tree = build.spec.tree
	self.mustHave = copyTable(options.mustHave or { }, true)
	self.mustNot = copyTable(options.mustNot or { }, true)
	self.lockedSockets = self:PrepareJewelSockets()
	self.requirements = options.requirements or { }
	self.fromScratch = options.fromScratch
	self.powerStat = options.powerStat or { stat = "PowerMetric", label = "Power Metric" }
	self.maxPathLength = options.maxPathLength or 4
	self.maxBranchSize = options.maxBranchSize or 6
	self.timeLimit = options.timeLimit or 60
	self.maxEvals = options.maxEvals or math.huge
	self.evalClock = options.clock == "evals"
	self.evalTime = options.evalTime or DEFAULT_EVAL_TIME
	-- Shared by reference with the model search's proxy, which reads and charges the same clock
	self.virtualClock = { extra = 0 }
	-- A list when set: how well the model's predictions matched the calculations that checked them
	self.predictionLog = options.logPredictions and { } or nil
	self.onProgress = options.onProgress
	self.shouldStop = options.shouldStop
	self.useFullDPS = true
	self.useModel = options.useModel ~= false
	self.modelShare = options.modelShare
	self.modelMeasureShare = options.modelMeasureShare
	-- Keep the values the model needs from every calculation, so trees the search has calculated anyway (e.g. a
	-- branch removed by Refine) serve as model measurements for free
	self.captureFields = self.useModel
	-- Structured model: the fields the score function reads are discovered from its first evaluations, and a tree's
	-- stat is predicted by running the score function on a predicted output (see DiscoverFields)
	self.structured = self.useModel and options.structuredModel == true
	-- What the model's checks have shown about its predictions (see RecordCheck)
	self.distancePenalty = self.useModel and options.distancePenalty ~= false
	self.calib = { buckets = { { n = 0, sum = 0 }, { n = 0, sum = 0 }, { n = 0, sum = 0 }, { n = 0, sum = 0 } },
		curve = { { 0, 0 } }, n = 0, plainErr = 0, structErr = 0 }
	self.scoreFunction = options.scoreFunction
	self.discRuns = 0
	self.discPaths, self.discSet, self.discParts = { }, { }, { }
	if options.scoreFunction then
		self.scoreFunc = options.scoreFunction
	elseif self.powerStat.stat == "PowerMetric" then
		local calcs = build.calcsTab.calcs
		self.scoreFunc = function(output)
			return calcs.powerMetricFromOutput(output) or 0
		end
	else
		self.scoreFunc = function(output)
			return data.powerStatList.GetFromOutput(output, self.powerStat)
		end
	end
	-- Seeded generator so runs are reproducible
	local seed = options.seed or 1
	self.random = function(lo, hi)
		-- Park-Miller; products stay below 2^53 so doubles are exact
		seed = (seed * 16807) % 2147483647
		return lo + seed % (hi - lo + 1)
	end

	self.root = self.spec.nodes[self.spec.curClass.startNodeId]
	self.calcFunc, self.baseOutput = build.calcsTab:GetMiscCalculator()
	self.skillCosts = self:CollectSkillCosts()
	self.sustainCost = self.requirements.SustainCost ~= 0 and TreeOptimizer.MainSkillUsedContinuously(build)

	local used = self.spec:CountAllocNodes()
	self.points = options.points or used

	self.evalCount = 0
	self.cacheHits = 0
	self.scoreCache = { }
	self.effectNodes = { }
	self.phase = "init"
	self.stopped = false
	return self
end

-- Every time read in the optimizer goes through here: real time (the app; CPU time headless) unless the evals clock is on.
function TreeOptimizer:Now()
	if self.evalClock then
		local clock = self.virtualClock
		clock.extra = clock.extra + CLOCK_READ_COST
		return (self.evalCount + clock.extra) * self.evalTime
	end
	return now()
end

-- Jewel sockets are ordinary nodes that carry the jewel socketed in them: the calculator applies a jewel exactly when
-- its socket is in the tree, and a cluster jewel's nodes hang off its socket. A few jewels change the tree itself,
-- which the spec only works out when it rebuilds paths, so they get extra handling:
--   * Timeless jewels conquer the nodes in their radius. When the socket is dropped, those nodes are evaluated with
--     their original stats (see conqueredBySocket). Abyss timeless jewels conquer along a path instead, so their
--     sockets stay allocated.
--   * Split Personality scales with the distance to the class start, which is recomputed for every candidate tree.
--   * Intuitive Leap-likes make nodes in their radius count as connected while the socket is allocated.
-- Unallocated sockets are left out when their jewel is also socketed in an allocated socket (a jewel can't count
-- twice), or when their jewel changes the tree, since its effect isn't known until the spec allocates the socket.
-- Returns [socketId] = reason for every socket the optimizer may not freely add or remove.
function TreeOptimizer:PrepareJewelSockets()
	local locked = { }
	-- [nodeId] = socketId: the node's stats in the spec come from the timeless jewel in that socket
	self.conqueredBySocket = { }
	-- [socketId] = true for Split Personality-like jewels
	self.distanceScaledSockets = { }
	-- Nodes that an Intuitive Leap-like jewel lets you allocate without a path
	self.leapNodes = { }

	local allocatedItems = { }
	for socketId, itemId in pairs(self.spec.jewels) do
		if self.spec.allocNodes[socketId] then
			allocatedItems[itemId] = true
		end
	end
	for socketId, itemId in pairs(self.spec.jewels) do
		local node = self.spec.nodes[socketId]
		local item = self.build.itemsTab.items[itemId]
		local jewelData = item and item.jewelData
		if node and item and not node.ascendancyName then
			local changesTree = jewelData and (jewelData.conqueredBy or jewelData.intuitiveLeapLike
				or (jewelData.impossibleEscapeKeystones and next(jewelData.impossibleEscapeKeystones)))
			if not self.spec.allocNodes[socketId] then
				if allocatedItems[itemId] then
					self.mustNot[socketId] = true
					locked[socketId] = item.name .. " is already socketed elsewhere"
				elseif changesTree then
					self.mustNot[socketId] = true
					locked[socketId] = item.name .. " changes the tree and its socket isn't allocated"
				end
			elseif jewelData then
				local conqueredBy = jewelData.conqueredBy
				if conqueredBy then
					if tostring(conqueredBy.conqueror and conqueredBy.conqueror.type):match("^abyss") or not item.jewelRadiusIndex then
						self.mustHave[socketId] = self.mustHave[socketId] or true
						locked[socketId] = item.name .. " conquers nodes along a path"
					elseif node.nodesInRadius and node.nodesInRadius[item.jewelRadiusIndex] then
						for nodeId in pairs(node.nodesInRadius[item.jewelRadiusIndex]) do
							local radiusNode = self.spec.nodes[nodeId]
							if radiusNode and radiusNode.conqueredBy and self.tree.nodes[nodeId] and not self.spec.hashOverrides[nodeId] then
								self.conqueredBySocket[nodeId] = socketId
							end
						end
					end
				end
				if jewelData.jewelIncEffectFromClassStart then
					self.distanceScaledSockets[socketId] = true
				end
			end
		end
	end
	for _, node in pairs(self.spec.nodes) do
		if node.intuitiveLeapLikesAffecting and node.intuitiveLeapLikesAffecting[1] then
			t_insert(self.leapNodes, node)
		end
	end
	return locked
end

-- The node object to use in calculations for node `id` of `state`: nodes conquered by a timeless jewel whose socket
-- isn't in the state revert to their original stats
function TreeOptimizer:CalcNode(state, id, node)
	local socketId = self.conqueredBySocket[id]
	if socketId and not state.nodes[socketId] then
		return self.tree.nodes[id]
	end
	return node
end

-- Number of allocated passives between a socket and the class start, as PassiveSpec:SetNodeDistanceToClassStart
function TreeOptimizer:DistanceToClassStart(state, socketId)
	local socket = self.spec.nodes[socketId]
	local dist = { [socketId] = 0 }
	local queue = { socket }
	local head = 1
	while queue[head] do
		local node = queue[head]
		head = head + 1
		for _, other in ipairs(node.linked) do
			if other == self.root then
				return dist[node.id]
			end
			if state.nodes[other.id] and canLeave(node) and not dist[other.id] then
				dist[other.id] = dist[node.id] + 1
				t_insert(queue, other)
			end
		end
	end
	return 0
end

-- Graph helpers

-- Nodes the optimizer may allocate: main tree nodes (and existing cluster nodes), never ascendancy or start nodes
function TreeOptimizer:IsEligible(node)
	if not node or node.ascendancyName or node.isProxy or (node.group and node.group.isProxy) then
		return false
	end
	if node.type == "ClassStart" or node.type == "AscendClassStart" then
		return false
	end
	if self.mustNot[node.id] then
		return false
	end
	-- Mastery nodes need an effect to be worth anything
	if node.type == "Mastery" and not self:GetMasteryEffects(node)[1] and not self.spec.allocNodes[node.id] then
		return false
	end
	return true
end

-- Returns the set of nodes in `set` that are connected to the class start, walking only through `set`, and the subset
-- connected by a path. Nodes in the radius of an Intuitive Leap-like jewel only need its socket to be connected, but
-- like in PassiveSpec:BuildAllDependsAndPaths, paths can't continue from them.
function TreeOptimizer:Reachable(set, without)
	local reached = { }
	local queue = { self.root }
	local head = 1
	while queue[head] do
		local node = queue[head]
		head = head + 1
		if canLeave(node) then
			for _, other in ipairs(node.linked) do
				local id = other.id
				if set[id] and not reached[id] and id ~= without then
					reached[id] = true
					t_insert(queue, other)
				end
			end
		end
	end
	local pathed = reached
	if self.leapNodes[1] then
		pathed = copyTable(reached, true)
		for _, node in ipairs(self.leapNodes) do
			local id = node.id
			if set[id] and not reached[id] and id ~= without then
				for _, socket in ipairs(node.intuitiveLeapLikesAffecting) do
					if pathed[socket.id] then
						reached[id] = true
						break
					end
				end
			end
		end
	end
	return reached, pathed
end

-- Multi-source BFS from the tree, through eligible unallocated nodes.
-- Returns dist[id] and parent[id] so a path back into the tree can be rebuilt.
function TreeOptimizer:Frontier(state, maxDist, banned)
	local dist, parent = { }, { }
	local queue = { self.root }
	local _, pathed = self:Reachable(state.nodes)
	for id in pairs(state.nodes) do
		if pathed[id] then
			t_insert(queue, self.spec.nodes[id])
		end
	end
	-- Nodes an Intuitive Leap-like jewel in the tree lets you allocate on their own
	if maxDist >= 1 then
		for _, node in ipairs(self.leapNodes) do
			local id = node.id
			if not state.nodes[id] and self:IsEligible(node) and not (banned and banned[id]) then
				for _, socket in ipairs(node.intuitiveLeapLikesAffecting) do
					if pathed[socket.id] then
						dist[id] = 1
						break
					end
				end
			end
		end
	end
	local head = 1
	while queue[head] do
		local node = queue[head]
		head = head + 1
		local d = dist[node.id] or 0
		if d < maxDist and canLeave(node) then
			for _, other in ipairs(node.linked) do
				local id = other.id
				if not dist[id] and not state.nodes[id] and other ~= self.root and self:IsEligible(other) and not (banned and banned[id]) then
					dist[id] = d + 1
					parent[id] = node
					t_insert(queue, other)
				end
			end
		end
	end
	return dist, parent
end

-- Nodes on the path from `node` back to the tree (excluding nodes already in the tree)
local function buildPath(node, dist, parent)
	local path = { }
	while node and dist[node.id] do
		t_insert(path, node)
		node = parent[node.id]
	end
	return path
end

-- Path to `node` (target first, as buildPath returns it). A node conquered by a timeless jewel only has its conquered stats
-- while the jewel's socket is in the tree (see CalcNode), so the path to the socket comes with it.
function TreeOptimizer:TargetPath(state, node, dist, parent)
	local path = buildPath(node, dist, parent)
	local socketId = self.conqueredBySocket[node.id]
	if socketId and not state.nodes[socketId] and dist[socketId] then
		local onPath = { }
		for _, other in ipairs(path) do
			onPath[other.id] = true
		end
		for _, other in ipairs(buildPath(self.spec.nodes[socketId], dist, parent)) do
			if not onPath[other.id] then
				t_insert(path, other)
				onPath[other.id] = true
			end
		end
	end
	return path
end

-- Masteries

function TreeOptimizer:GetMasteryEffects(node)
	local treeNode = self.tree.nodes[node.id]
	return treeNode and treeNode.masteryEffects or { }
end

-- A stand-in node carrying a single mastery effect's stats, used to evaluate unallocated masteries
function TreeOptimizer:GetEffectNode(node, effectId)
	local specNode = self.spec.allocNodes[node.id]
	if specNode and self.spec.masterySelections[node.id] == effectId then
		return specNode
	end
	local key = node.id .. ":" .. effectId
	if not self.effectNodes[key] then
		local effect = self.tree.masteryEffects[effectId]
		local effectNode = { id = node.id, type = node.type, name = node.name, dn = node.dn, sd = { } }
		for i, sd in ipairs(effect and effect.sd or { }) do
			effectNode.sd[i] = sd
		end
		self.tree:ProcessStats(effectNode)
		self.effectNodes[key] = effectNode
	end
	return self.effectNodes[key]
end

-- Effects of `node` that no other mastery in the state uses
function TreeOptimizer:AvailableEffects(state, node)
	local used = { }
	for id, effectId in pairs(state.mastery) do
		if id ~= node.id then
			used[effectId] = true
		end
	end
	local list = { }
	for _, effect in ipairs(self:GetMasteryEffects(node)) do
		if not used[effect.effect] then
			t_insert(list, effect.effect)
		end
	end
	return list
end

-- State: { nodes = { [id] = node object used in calcs }, mastery = { [id] = effect id }, count = points used }

function TreeOptimizer:NewState()
	return { nodes = { }, mastery = { }, count = 0 }
end

function TreeOptimizer:CopyState(state)
	local copy = self:NewState()
	for id, node in pairs(state.nodes) do
		copy.nodes[id] = node
	end
	for id, effectId in pairs(state.mastery) do
		copy.mastery[id] = effectId
	end
	copy.count = state.count
	copy.eval = state.eval
	return copy
end

-- Adds `node` to the state. Mastery nodes take an effect id.
function TreeOptimizer:AddNode(state, node, effectId)
	if state.nodes[node.id] then
		return
	end
	if node.type == "Mastery" then
		effectId = effectId or self.spec.masterySelections[node.id]
		state.mastery[node.id] = effectId
		state.nodes[node.id] = self:GetEffectNode(node, effectId)
	else
		state.nodes[node.id] = node
	end
	state.count = state.count + 1
	state.eval = nil
end

function TreeOptimizer:RemoveNode(state, id)
	if state.nodes[id] then
		state.nodes[id] = nil
		state.mastery[id] = nil
		state.count = state.count - 1
		state.eval = nil
	end
end

-- Removes every node no longer connected to the class start
function TreeOptimizer:Prune(state)
	local reached = self:Reachable(state.nodes)
	for id in pairs(state.nodes) do
		if not reached[id] then
			self:RemoveNode(state, id)
		end
	end
end

-- Evaluation

local function stateKey(state)
	local keys = { }
	for id in pairs(state.nodes) do
		t_insert(keys, state.mastery[id] and (id .. ":" .. state.mastery[id]) or id)
	end
	t_sort(keys, function(a, b) return tostring(a) < tostring(b) end)
	return t_concat(keys, ",")
end

-- Deficit added per skill that can't pay its cost, on top of the missing amount
local COST_WARNING_DEFICIT = 20

-- Records what every active skill costs in the current build. The misc calculator only processes the main skill,
-- so per-skill costs come from the main calculation's cache and are assumed not to change with the tree.
-- Mirrors the cost warning checks in calcs.buildOutput. Skills that can't already pay their cost are ignored.
function TreeOptimizer:CollectSkillCosts()
	local costs = { }
	local env = self.build.calcsTab.mainEnv
	local cache = GlobalCache.cachedData["MAIN"]
	if not env or not cache then
		return costs
	end
	local output = env.player.output
	local esProtectsMana = output.EnergyShieldProtectsMana
	local pools = {
		{ pool = "LifeUnreserved", cost = "LifeCost", raw = "Life" },
		{ pool = "ManaUnreserved", cost = "ManaCost", raw = "Mana" },
		{ pool = "Rage", cost = "RageCost" },
		{ pool = "EnergyShield", cost = "ESCost" },
		{ pool = "LifeUnreservedPercent", cost = "LifePercentCost" },
		{ pool = "ManaUnreservedPercent", cost = "ManaPercentCost" },
	}
	for _, skill in ipairs(env.player.activeSkillList) do
		local entry = cache[cacheSkillUUID(skill, env)]
		if entry and entry.Env then
			local skillOutput = entry.Env.player.output
			local mainSkill = entry.Env.player.mainSkill
			for _, p in ipairs(pools) do
				local cost = skillOutput[p.cost]
				local reserves = p.raw and mainSkill and mainSkill.skillData[p.raw .. "ReservedPercent"]
				if cost and cost > 0 and not reserves then
					local entryCost = {
						name = skill.activeEffect.grantedEffect.name,
						pool = p.pool,
						cost = cost,
						addES = esProtectsMana and p.cost == "ManaCost",
					}
					if self:SkillCostShortfall(entryCost, output) <= 0 then
						t_insert(costs, entryCost)
					end
				end
			end
		end
	end
	return costs
end

function TreeOptimizer:SkillCostShortfall(skillCost, output)
	local pool = (output[skillCost.pool] or 0) + (skillCost.addES and (output.EnergyShield or 0) or 0)
	return skillCost.cost - pool
end

-- Cost per second assumes the skill is used back to back. That holds for attacks, spells, triggers, traps and mines,
-- but summons, totems and brands are placed once and left to work, so their cost per second is meaningless.
function TreeOptimizer.MainSkillUsedContinuously(build)
	local env = build.calcsTab.mainEnv
	local mainSkill = env and env.player.mainSkill
	if not mainSkill then
		return false
	end
	local flags = mainSkill.skillFlags or { }
	return not (flags.minion or flags.totem or (mainSkill.skillTypes and mainSkill.skillTypes[SkillType.Brand]))
end

-- Deficit per percent of the main skill's cost per second that recovery doesn't cover
local SUSTAIN_DEFICIT_PER_PERCENT = 1

-- Recovery per second of a resource while using the main skill non-stop: regeneration and flat recovery (net of
-- degens), plus leech and gain on hit. Energy shield recharge counts too, since spending it doesn't interrupt recharge.
local function recoveryRate(output, resource)
	local regen = output["Net" .. resource .. "Regen"] or output[resource .. "RegenRecovery"] or 0
	local recovery = regen + (output[resource .. "LeechGainRate"] or 0)
	if resource == "EnergyShield" then
		recovery = recovery + (output.EnergyShieldRecharge or 0)
	end
	return recovery
end

local function costPerSecond(output, costPrefix, pool)
	return (output[costPrefix .. "PerSecondCost"] or 0) + (output[costPrefix .. "PercentPerSecondCost"] or 0) * (output[pool] or 0) / 100
end

-- Each resource the main skill spends per second, with the recovery that pays for it.
-- With Eldritch Battery mana costs are paid from energy shield first and mana after, so both pools share the cost.
function TreeOptimizer.SustainShortfalls(output)
	local list = { }
	local function check(label, cost, recovery)
		if cost > 0 and recovery < cost then
			t_insert(list, { label = label, cost = cost, recovery = recovery })
		end
	end
	local manaCost = costPerSecond(output, "Mana", "Mana")
	local esCost = costPerSecond(output, "ES", "EnergyShield")
	if output.EnergyShieldProtectsMana then
		check("mana/ES", manaCost + esCost, recoveryRate(output, "Mana") + recoveryRate(output, "EnergyShield"))
	else
		check("mana", manaCost, recoveryRate(output, "Mana"))
		check("ES", esCost, recoveryRate(output, "EnergyShield"))
	end
	check("life", costPerSecond(output, "Life", "Life"), recoveryRate(output, "Life"))
	return list
end

-- Hard constraints, as a single "deficit" (0 = all met) plus details for display:
--   attributes meet requirements (or the manual overrides), unreserved life >= 1,
--   unreserved mana >= the minimum (0 unless set), every skill can pay its cost, and recovery keeps up with the
--   main skill's cost per second (unless turned off)
function TreeOptimizer:ConstraintDeficit(output)
	local deficit = 0
	local attributes = { }
	-- Substrings of mod names that can help with the violated constraints (see Repair)
	local needs = { }
	local function need(...)
		for _, pattern in ipairs({ ... }) do
			needs[pattern] = true
		end
	end
	local omni = output.ReqOmni and output.ReqOmni > 0
	for _, attr in ipairs(omni and { "Omni" } or { "Str", "Dex", "Int" }) do
		local req = self.requirements[attr] or output["Req" .. attr] or 0
		local have = output[attr] or 0
		attributes[attr] = { req = req, have = have }
		if have < req then
			deficit = deficit + (req - have)
			need(attr, "Attribute")
		end
	end
	local violations = { }
	if output.LifeUnreserved and output.LifeUnreserved < 1 and not output.ChaosInoculation then
		deficit = deficit + (1 - output.LifeUnreserved) * 0.5
		t_insert(violations, "unreserved life below 1")
		need("Life", "Reserv")
	end
	local minMana = self.requirements.ManaUnreserved or 0
	if output.ManaUnreserved and output.ManaUnreserved < minMana then
		deficit = deficit + (minMana - output.ManaUnreserved) * 0.5
		t_insert(violations, minMana > 0 and ("unreserved mana below " .. minMana) or "unreserved mana negative")
		need("Mana", "Reserv")
	end
	for _, skillCost in ipairs(self.skillCosts) do
		local shortfall = self:SkillCostShortfall(skillCost, output)
		if shortfall > 0 then
			deficit = deficit + COST_WARNING_DEFICIT + shortfall * 0.5
			t_insert(violations, "can't pay " .. skillCost.pool:gsub("Unreserved", "") .. " cost of " .. skillCost.name)
			need((skillCost.pool:gsub("Unreserved", ""):gsub("Percent", "")), "Reserv", "Cost")
			if skillCost.addES then
				need("EnergyShield")
			end
		end
	end
	if self.sustainCost then
		for _, shortfall in ipairs(TreeOptimizer.SustainShortfalls(output)) do
			deficit = deficit + (1 - shortfall.recovery / shortfall.cost) * 100 * SUSTAIN_DEFICIT_PER_PERCENT
			t_insert(violations, string.format("main skill costs %.1f %s/s, recovering %.1f/s", shortfall.cost, shortfall.label, shortfall.recovery))
			need("Cost")
			for resource, label in pairs({ Mana = "mana", EnergyShield = "ES", Life = "life" }) do
				if shortfall.label:find(label, 1, true) then
					need(resource)
				end
			end
		end
	end
	return deficit, attributes, violations, needs
end

-- Everything ConstraintDeficit reads, so the node value model can predict constraints from summed node effects
local CONSTRAINT_FIELDS = { "Str", "Dex", "Int", "Omni", "ReqStr", "ReqDex", "ReqInt", "ReqOmni", "Life", "LifeUnreserved",
	"LifeUnreservedPercent", "Mana", "ManaUnreserved", "ManaUnreservedPercent", "EnergyShield", "Rage",
	"ManaPerSecondCost", "ManaPercentPerSecondCost", "ESPerSecondCost", "ESPercentPerSecondCost", "LifePerSecondCost",
	"LifePercentPerSecondCost", "NetManaRegen", "ManaRegenRecovery", "ManaLeechGainRate", "NetEnergyShieldRegen",
	"EnergyShieldRegenRecovery", "EnergyShieldLeechGainRate", "EnergyShieldRecharge", "NetLifeRegen", "LifeRegenRecovery",
	"LifeLeechGainRate" }
local CONSTRAINT_FLAGS = { "ChaosInoculation", "EnergyShieldProtectsMana" }

-- A table that logs every key read from `t` into `reads` (path -> type of the value), with nested tables proxied
-- too. Arrays are returned as they are (a proxy can't answer `#`): the formula reading only their length is
-- recorded as a read of the array, which the model treats as constant.
local function recordingProxy(t, prefix, reads)
	return setmetatable({ }, { __index = function(_, key)
		local value = t[key]
		local path = prefix .. tostring(key)
		reads[path] = type(value)
		if type(value) == "table" and #value == 0 then
			return recordingProxy(value, path .. ".", reads)
		end
		return value
	end })
end

local function getPath(output, parts)
	local value = output
	for _, key in ipairs(parts) do
		if type(value) ~= "table" then
			return nil
		end
		value = value[key]
	end
	return value
end

-- Number of the first evaluations whose score function reads are recorded (branches can read different fields)
local DISCOVERY_RUNS = 3

-- Records which numeric output fields the score function reads on `output`, adding them to the modelled fields. If
-- the function doesn't reproduce `value` on the proxy, or reads no number, the structured model is switched off.
function TreeOptimizer:DiscoverFields(output, value)
	self.discRuns = self.discRuns + 1
	local reads = { }
	local ok, result = pcall(self.scoreFunc, recordingProxy(output, "", reads))
	if not ok or type(result) ~= "number" or math.abs(result - value) > 1e-9 * math.max(1, math.abs(value)) then
		self.structured = false
		return
	end
	local found = { }
	for path, kind in pairs(reads) do
		if kind == "number" and not self.discSet[path] then
			t_insert(found, path)
		end
	end
	t_sort(found)
	for _, path in ipairs(found) do
		self.discSet[path] = true
		t_insert(self.discPaths, path)
		local parts = { }
		for part in path:gmatch("[^.]+") do
			t_insert(parts, part)
		end
		self.discParts[path] = parts
	end
	if self.discRuns == 1 then
		self.template = output
	end
	if self.discRuns >= DISCOVERY_RUNS and #self.discPaths == 0 then
		self.structured = false
	end
end

-- The discovered fields' values in a captured field table (for the prediction log)
function TreeOptimizer:DiscoveredValues(fields)
	if not (self.structured and fields) then
		return nil
	end
	local values = { }
	for _, path in ipairs(self.discPaths) do
		values[path] = fields[path]
	end
	return values
end

function TreeOptimizer:ScoreFromOutput(output)
	local value
	if self.scoreFunction then
		value = self.scoreFunction(output)
	else
		value = data.powerStatList.GetFromOutput(output, self.powerStat)
	end
	if self.structured and self.discRuns < DISCOVERY_RUNS then
		self:DiscoverFields(output, value)
	end
	local deficit, attributes, violations, needs = self:ConstraintDeficit(output)
	-- Work in log space so gains are relative (a 2% increase is worth the same anywhere)
	local logValue = value > 0 and m_log(value) or (-100 + value * EPSILON)
	local fields
	if self.captureFields then
		fields = { logValue = logValue }
		for _, key in ipairs(CONSTRAINT_FIELDS) do
			fields[key] = output[key]
		end
		for _, key in ipairs(CONSTRAINT_FLAGS) do
			fields[key] = output[key]
		end
		if self.structured then
			for _, path in ipairs(self.discPaths) do
				local v = getPath(output, self.discParts[path])
				if type(v) == "number" and v == v and v ~= math.huge and v ~= -math.huge then
					fields[path] = v
				end
			end
		end
	end
	return {
		value = value,
		deficit = deficit,
		attributes = attributes,
		violations = violations,
		needs = needs,
		score = logValue - deficit * ATTRIBUTE_PENALTY,
		fields = fields,
	}
end

function TreeOptimizer:Evaluate(state)
	if state.eval then
		return state.eval
	end
	local key = stateKey(state)
	local cached = self.scoreCache[key]
	if cached then
		self.cacheHits = self.cacheHits + 1
		state.eval = cached
		return cached
	end

	local addNodes, removeNodes, keep = { }, { }, { }
	for id, node in pairs(state.nodes) do
		local calcNode = self:CalcNode(state, id, node)
		if self.spec.allocNodes[id] == calcNode then
			keep[id] = true
		else
			addNodes[calcNode] = true
		end
	end
	for id, node in pairs(self.spec.allocNodes) do
		if not node.ascendancyName and node.type ~= "ClassStart" and node.type ~= "AscendClassStart" and not keep[id] then
			removeNodes[node] = true
		end
	end
	-- The calculator reads Split Personality's distance from the spec node, so set it for this tree while calculating
	local savedDistances = { }
	for socketId in pairs(self.distanceScaledSockets) do
		if state.nodes[socketId] then
			local socket = self.spec.nodes[socketId]
			t_insert(savedDistances, { socket, socket.distanceToClassStart })
			socket.distanceToClassStart = self:DistanceToClassStart(state, socketId)
		end
	end
	-- Radius jewels read unallocated nodes from the spec (e.g. Unnatural Instinct), so nodes conquered by a timeless
	-- jewel that isn't in this tree are swapped for their originals while calculating
	local swappedNodes = { }
	for id, socketId in pairs(self.conqueredBySocket) do
		if not state.nodes[socketId] then
			t_insert(swappedNodes, { id, self.spec.nodes[id] })
			self.spec.nodes[id] = self.tree.nodes[id]
		end
	end
	local ok, output = pcall(self.calcFunc, { addNodes = addNodes, removeNodes = removeNodes }, self.useFullDPS)
	for _, saved in ipairs(savedDistances) do
		saved[1].distanceToClassStart = saved[2]
	end
	for _, swapped in ipairs(swappedNodes) do
		self.spec.nodes[swapped[1]] = swapped[2]
	end
	if not ok then
		error(output, 0)
	end
	local result = self:ScoreFromOutput(output)
	self.scoreCache[key] = result
	state.eval = result
	self.evalCount = self.evalCount + 1
	if self.onProgress then
		self.onProgress({ phase = self.phase, evals = self.evalCount, best = self.best and self.best.eval, elapsed = self:Now() - (self.startTime or self:Now()), timeLimit = self.timeLimit })
	end
	if (self.shouldStop and self.shouldStop()) or self.evalCount >= self.maxEvals then
		self.stopped = true
	end
	-- Past the time limit, every search loop winds down the same way as when stopped
	if not self.stopped and self.deadline and self:Now() >= self.deadline then
		self.stopped = true
		self.timeUp = true
	end
	return result
end

-- Score that weighs the constraint deficit heavily, for choosing between trees that don't meet the requirements: a tree
-- that is closer to meeting them is better unless the other one is much stronger. Repair ranks its moves by it.
function TreeOptimizer:Merit(eval)
	if eval.deficit == 0 then
		return eval.score
	end
	return eval.score - eval.deficit * INFEASIBLE_PENALTY
end

-- Feasible trees always beat infeasible ones. Feasible trees compare by score, infeasible ones by merit.
function TreeOptimizer:IsBetter(a, b)
	if (a.deficit == 0) ~= (b.deficit == 0) then
		return a.deficit == 0
	end
	if a.deficit == 0 then
		return a.score > b.score + EPSILON
	end
	return self:Merit(a) > self:Merit(b) + EPSILON
end

function TreeOptimizer:Track(state)
	local eval = self:Evaluate(state)
	if state.count <= self.points and (not self.best or self:IsBetter(eval, self.best.eval)) then
		self.best = self:CopyState(state)
	end
end

-- Phase 2: connect must-haves

function TreeOptimizer:ConnectMustHaves(state)
	local pending = { }
	for id, value in pairs(self.mustHave) do
		if not state.nodes[id] then
			pending[id] = value
		end
	end
	while next(pending) do
		local dist, parent = self:Frontier(state, math.huge)
		local bestId
		for id in pairs(pending) do
			if dist[id] and (not bestId or dist[id] < dist[bestId]) then
				bestId = id
			end
		end
		if not bestId then
			local names = { }
			for id in pairs(pending) do
				t_insert(names, self.spec.nodes[id] and self.spec.nodes[id].dn or tostring(id))
			end
			return false, "Cannot reach must-have node(s): " .. t_concat(names, ", ")
		end
		local target = self.spec.nodes[bestId]
		local path = buildPath(target, dist, parent)
		for i = #path, 2, -1 do
			self:AddNode(state, path[i])
		end
		if target.type == "Mastery" then
			local effectId = type(pending[bestId]) == "number" and pending[bestId] or self.spec.masterySelections[bestId]
			if not effectId then
				effectId = self:BestMasteryEffect(state, target)
			end
			self:AddNode(state, target, effectId)
		else
			self:AddNode(state, target)
		end
		pending[bestId] = nil
	end
	return true
end

-- A mastery's effects are ranked with one calculation each. Which effect is best rarely changes while the tree grows or
-- shifts by a few nodes, so until the tree has changed size by this fraction since the last full ranking, only the
-- MASTERY_TOP_EFFECTS best effects of that ranking are evaluated again. Masteries of the same kind (e.g. every Life
-- Mastery) offer the same effects, so they share a ranking.
local MASTERY_RERANK_CHANGE = 0.2
local MASTERY_TOP_EFFECTS = 2

function TreeOptimizer:MasteryRankKey(node)
	local keys = self:OwnTable("masteryRankKeys")
	if not keys[node.id] then
		local ids = { }
		for _, effect in ipairs(self:GetMasteryEffects(node)) do
			t_insert(ids, effect.effect)
		end
		keys[node.id] = t_concat(ids, ",")
	end
	return keys[node.id]
end

-- Whether this object scores trees with predictions: ModelRound's proxy inherits from an optimizer instead of from
-- TreeOptimizer. Predictions cost microseconds, so the shortcuts that save full calculations would only make the search
-- on the model worse.
function TreeOptimizer:PredictsScores()
	return getmetatable(self) ~= TreeOptimizer
end

-- A cache table that belongs to this object. The model search (ModelRound's proxy) inherits from the optimizer, and
-- caches filled from predictions must not leak into the real search, or the other way round.
function TreeOptimizer:OwnTable(name)
	local tbl = rawget(self, name)
	if not tbl then
		tbl = { }
		rawset(self, name, tbl)
	end
	return tbl
end

function TreeOptimizer:BestMasteryEffect(state, node)
	local ranks = self:OwnTable("masteryRanks")
	local available = self:AvailableEffects(state, node)
	local list = available
	local rankKey = self:MasteryRankKey(node)
	local rank = not self:PredictsScores() and ranks[rankKey]
	if rank and math.abs(state.count - rank.count) <= rank.count * MASTERY_RERANK_CHANGE then
		local isAvailable = { }
		for _, effectId in ipairs(available) do
			isAvailable[effectId] = true
		end
		list = { }
		for _, effectId in ipairs(rank.order) do
			if isAvailable[effectId] then
				t_insert(list, effectId)
				if #list >= MASTERY_TOP_EFFECTS then
					break
				end
			end
		end
		if not list[1] then
			list = available
		end
	end
	local bestEffect, bestEval
	local evals = { }
	for _, effectId in ipairs(list) do
		local trial = self:CopyState(state)
		self:AddNode(trial, node, effectId)
		local eval = self:Evaluate(trial)
		evals[effectId] = eval
		if not bestEval or self:IsBetter(eval, bestEval) then
			bestEffect, bestEval = effectId, eval
		end
	end
	if list == available and available[1] then
		local order = copyTable(available, true)
		t_sort(order, function(a, b)
			local ea, eb = evals[a], evals[b]
			if (ea.deficit == 0) ~= (eb.deficit == 0) then
				return ea.deficit == 0
			end
			if ea.score ~= eb.score then
				return ea.score > eb.score
			end
			return a < b
		end)
		ranks[rankKey] = { count = state.count, order = order }
	end
	return bestEffect, bestEval
end

-- Branches: removing a node also removes everything only connected through it

-- What removing each node cuts off, from one depth-first pass over the tree instead of one search per node (Tarjan's
-- articulation points): removing a node cuts off itself and each of its DFS subtrees with no link to a node above it.
-- Masteries are left out of the pass, since paths can't continue from them; one is cut off when all of its allocated
-- neighbours are. Returns nil when the pass doesn't apply (Intuitive Leap-likes, or a tree not fully connected).
function TreeOptimizer:BranchCuts(state)
	if self.leapNodes[1] then
		return nil
	end
	local nodes = state.nodes
	local specNodes = self.spec.nodes
	local rootId = self.root.id
	local disc, low, parent, children = { [rootId] = 0 }, { [rootId] = 0 }, { }, { [rootId] = { } }
	local finished = { }
	local count = 0
	local stack = { { node = self.root, i = 1 } }
	while stack[1] do
		local top = stack[#stack]
		local node = top.node
		local other = node.linked[top.i]
		if other then
			top.i = top.i + 1
			local id = other.id
			-- The class start counts as a link to the tree: a node next to it stays connected when another path to it is cut
			if (nodes[id] or id == rootId) and other.type ~= "Mastery" then
				if not disc[id] then
					count = count + 1
					disc[id], low[id], parent[id], children[id] = count, count, node.id, { }
					t_insert(children[node.id], id)
					t_insert(stack, { node = other, i = 1 })
				elseif id ~= parent[node.id] and disc[id] < low[node.id] then
					low[node.id] = disc[id]
				end
			end
		else
			stack[#stack] = nil
			local p = parent[node.id]
			if p and low[node.id] < low[p] then
				low[p] = low[node.id]
			end
			t_insert(finished, node.id)
		end
	end
	local masteries = { }
	for id in pairs(nodes) do
		if specNodes[id].type == "Mastery" then
			-- Allocated neighbours a mastery hangs off
			local neighbours = { }
			for _, other in ipairs(specNodes[id].linked) do
				if disc[other.id] and other.id ~= rootId then
					t_insert(neighbours, other.id)
				end
			end
			if not neighbours[1] then
				return nil
			end
			masteries[id] = neighbours
		elseif not disc[id] then
			return nil
		end
	end
	-- Size and number of must-haves of each DFS subtree, children before parents
	local size, must = { }, { }
	for _, id in ipairs(finished) do
		if id ~= rootId then
			local s, m = 1, self.mustHave[id] and 1 or 0
			for _, child in ipairs(children[id]) do
				s, m = s + size[child], m + must[child]
			end
			size[id], must[id] = s, m
		end
	end
	return { disc = disc, low = low, children = children, size = size, must = must, masteries = masteries }
end

-- Returns a list of { ids = {...}, size = n } for every node whose branch can be removed
function TreeOptimizer:RemovableBranches(state, maxSize)
	local cuts = self:BranchCuts(state)
	if cuts then
		local branches = { }
		local disc, low, children, size, must = cuts.disc, cuts.low, cuts.children, cuts.size, cuts.must
		-- Masteries next to each node
		local masteriesNear = { }
		for masteryId, neighbours in pairs(cuts.masteries) do
			if not self.mustHave[masteryId] then
				t_insert(branches, { key = masteryId, ids = { masteryId }, size = 1 })
			end
			for _, id in ipairs(neighbours) do
				masteriesNear[id] = masteriesNear[id] or { }
				t_insert(masteriesNear[id], masteryId)
			end
		end
		for id in pairs(state.nodes) do
			if not self.mustHave[id] and not cuts.masteries[id] then
				local total, mustCut = 1, 0
				for _, child in ipairs(children[id]) do
					if low[child] >= disc[id] then
						total, mustCut = total + size[child], mustCut + must[child]
					end
				end
				if total <= maxSize and mustCut == 0 then
					local ids, inCut = { id }, { [id] = true }
					for _, child in ipairs(children[id]) do
						if low[child] >= disc[id] then
							local walk = { child }
							while walk[1] do
								local nodeId = walk[#walk]
								walk[#walk] = nil
								t_insert(ids, nodeId)
								inCut[nodeId] = true
								for _, grandchild in ipairs(children[nodeId]) do
									t_insert(walk, grandchild)
								end
							end
						end
					end
					-- Masteries cut off along with the branch
					local valid = true
					for i = 1, #ids do
						for _, masteryId in ipairs(masteriesNear[ids[i]] or { }) do
							if not inCut[masteryId] then
								local cut = true
								for _, neighbour in ipairs(cuts.masteries[masteryId]) do
									if not inCut[neighbour] then
										cut = false
										break
									end
								end
								if cut then
									if self.mustHave[masteryId] or #ids >= maxSize then
										valid = false
									end
									t_insert(ids, masteryId)
									inCut[masteryId] = true
								end
							end
						end
					end
					if valid then
						t_insert(branches, { key = id, ids = ids, size = #ids })
					end
				end
			end
		end
		t_sort(branches, function(a, b) return a.key < b.key end)
		return branches
	end
	local branches = { }
	for id in pairs(state.nodes) do
		if not self.mustHave[id] then
			local reached = self:Reachable(state.nodes, id)
			local ids = { id }
			local valid = true
			for otherId in pairs(state.nodes) do
				if otherId ~= id and not reached[otherId] then
					if self.mustHave[otherId] or #ids >= maxSize then
						valid = false
						break
					end
					t_insert(ids, otherId)
				end
			end
			if valid then
				t_insert(branches, { key = id, ids = ids, size = #ids })
			end
		end
	end
	-- Stable order so seeded runs are reproducible
	t_sort(branches, function(a, b) return a.key < b.key end)
	return branches
end

local function branchKey(state, branch)
	local keys = { }
	for _, id in ipairs(branch.ids) do
		t_insert(keys, id)
	end
	t_sort(keys)
	return t_concat(keys, ",")
end

-- Phase 3: trim to budget by removing the branch with the lowest loss per point

function TreeOptimizer:Trim(state, budget)
	while state.count > budget and not self.stopped do
		local base = self:Evaluate(state)
		local bestBranch, bestLoss
		for _, branch in ipairs(self:RemovableBranches(state, m_max(1, state.count - budget))) do
			local trial = self:CopyState(state)
			for _, id in ipairs(branch.ids) do
				self:RemoveNode(trial, id)
			end
			local eval = self:Evaluate(trial)
			local loss = (base.score - eval.score) / branch.size
			if base.deficit == 0 and eval.deficit > 0 then
				loss = loss + 1e3
			end
			if not bestLoss or loss < bestLoss then
				bestBranch, bestLoss = branch, loss
			end
		end
		if not bestBranch then
			-- Only branches longer than the overshoot exist; drop the cheapest single node branch available
			local branches = self:RemovableBranches(state, math.huge)
			t_sort(branches, function(a, b) return a.size < b.size end)
			bestBranch = branches[1]
			if not bestBranch then
				return false, "Must-have nodes need more points than the budget allows"
			end
		end
		for _, id in ipairs(bestBranch.ids) do
			self:RemoveNode(state, id)
		end
	end
	return state.count <= budget, state.count > budget and "Must-have nodes need more points than the budget allows" or nil
end

-- Furthest the fill reaches when nothing nearer improves the tree
local MAX_FILL_REACH = 10
-- A candidate whose estimate is out of date is ranked at this share of it, so a freshly evaluated candidate that is nearly as
-- good as the best estimate is taken without evaluating the next one (cuts the calculations per step by about a third)
local FILL_SLACK = 0.85
-- Furthest away a mastery is considered. Ranking a mastery's effects takes a calculation per effect, and the path to a
-- mastery runs through a notable that is a candidate of its own; once that is taken, the mastery is next to the tree.
local FILL_MASTERY_REACH = 1
-- During the first fill, how far away a plain travel node is evaluated as a target (0: not at all until the reach widens).
-- Most of the first fill's calculations go to measuring every node in reach, and the notables, masteries and keystones are
-- what pays; the travel nodes on the way are measured as part of the paths to them. This cuts the first fill by 30-40%
-- and leaves it better placed against the deadline; applying it to the later fills too made the trees worse.
local FILL_NORMAL_REACH = 0
-- Share of the time limit within which the thorough fill that follows a quick one has to finish (see Run)
local THOROUGH_FILL_SHARE = 0.5

-- Nodes with the same kind of stats (e.g. the many "+10 to Strength" nodes, or "5%" and "6% increased maximum Life")
-- are worth about the same, so a never-evaluated target takes the last measured gain per point of a target with its
-- signature as its estimate. The estimate only orders evaluations: nothing is taken before it's evaluated.
function TreeOptimizer:NodeSignature(node)
	local signatures = self:OwnTable("nodeSignatures")
	if not signatures[node.id] then
		if node.type == "Mastery" then
			signatures[node.id] = "mastery:" .. self:MasteryRankKey(node)
		elseif node.isJewelSocket or self.conqueredBySocket[node.id] or not node.sd then
			-- Sockets carry different jewels, and conquered nodes change with their socket
			signatures[node.id] = "node:" .. node.id
		else
			signatures[node.id] = node.type .. ":" .. t_concat(node.sd, "\n"):gsub("[%d%.]+", "#")
		end
	end
	return signatures[node.id]
end

-- Phase 4: lazy greedy fill
-- `estimates` persist between fills: [targetId] = { gain = score gain of the path when last evaluated, key = state key }.
-- Candidates are re-evaluated best estimate first, and the first one whose estimate is up to date for the current tree
-- is taken. Targets never evaluated estimate from a target with the same signature, or else go first.

function TreeOptimizer:Fill(state, budget, banned)
	self.estimates = self.estimates or { }
	local estimates = self.estimates
	-- [signature] = gain per point of the last evaluated target with that signature
	local signatureRates = self:OwnTable("signatureRates")
	local masteryReach = self:PredictsScores() and math.huge or FILL_MASTERY_REACH
	local useSignatures = not self:PredictsScores()
	local normalReach = not self:PredictsScores() and self.fillNormalReach or nil
	-- How far paths may reach. Widened when nothing within reach helps (e.g. the class start is surrounded by
	-- nodes that don't affect the stat) so points aren't left unspent. Narrowed again after every accepted path:
	-- a wide frontier brings in many never-evaluated nodes after each step, which costs far more than re-widening.
	local reach = self.maxPathLength
	local slack = FILL_SLACK
	while state.count < budget and not self.stopped do
		local base = self:Evaluate(state)
		local baseKey = stateKey(state)
		local remaining = budget - state.count
		local maxReach = m_min(m_max(self.maxPathLength, MAX_FILL_REACH), remaining)
		reach = m_min(reach, remaining)
		local dist, parent = self:Frontier(state, reach, banned)

		local candidates = { }
		for id, d in pairs(dist) do
			local node = self.spec.nodes[id]
			-- Plain travel nodes are mostly worth taking as part of the path to something else, so the first fill
			-- doesn't evaluate them as targets of their own until the reach widens (see FILL_NORMAL_REACH)
			local skipNormal = normalReach and node.type == "Normal" and d > normalReach and reach <= self.maxPathLength
				and not estimates[id] and not (useSignatures and signatureRates[self:NodeSignature(node)])
			if (node.type ~= "Mastery" or d <= masteryReach) and not skipNormal then
				local path
				if self.conqueredBySocket[id] then
					path = self:TargetPath(state, node, dist, parent)
					d = #path
				end
				local est = estimates[id]
				t_insert(candidates, { id = id, node = node, dist = d, path = path, priority = est and (est.gain / d), signature = self:NodeSignature(node), fresh = est and est.key == baseKey })
			end
		end
		local accepted
		while candidates[1] and not self.stopped do
			local best, bestPriority
			for _, cand in ipairs(candidates) do
				local priority = cand.priority or (useSignatures and signatureRates[cand.signature]) or math.huge
				if not cand.fresh and priority > 0 and priority < math.huge then
					priority = priority * slack
				end
				if not best or priority > bestPriority then
					best, bestPriority = cand, priority
				end
			end
			if best.fresh then
				accepted = best
				break
			end
			-- Re-evaluate the most promising candidate against the current tree
			local path = best.path or buildPath(best.node, dist, parent)
			local trial = self:CopyState(state)
			for i = #path, 2, -1 do
				self:AddNode(trial, path[i])
			end
			local eval, effectId
			if best.node.type == "Mastery" then
				effectId, eval = self:BestMasteryEffect(trial, best.node)
				if not effectId then
					eval = { score = -math.huge, deficit = math.huge }
				end
			else
				self:AddNode(trial, best.node)
				eval = self:Evaluate(trial)
			end
			local gain = eval.score - base.score
			if base.deficit == 0 and eval.deficit > 0 then
				gain = -1e3
			end
			estimates[best.id] = { gain = gain, key = baseKey, effectId = effectId }
			signatureRates[best.signature] = gain / best.dist
			best.priority = gain / best.dist
			best.fresh = true
			best.path = path
			best.effectId = effectId
		end
		if self.stopped then
			break
		end
		if not accepted or accepted.priority <= EPSILON then
			if reach >= maxReach then
				break
			end
			reach = m_min(reach * 2, maxReach)
			goto continue
		end
		reach = self.maxPathLength

		local path = accepted.path or buildPath(accepted.node, dist, parent)
		for i = #path, 2, -1 do
			self:AddNode(state, path[i])
		end
		self:AddNode(state, accepted.node, accepted.effectId or estimates[accepted.id].effectId)
		self:Track(state)
		::continue::
	end
	return state
end

-- Phase 5: refine by removing a branch and refilling without it. With `once`, returns after the first improving move
-- (so the caller can take its turn on the better tree), otherwise at a local optimum.

function TreeOptimizer:Refine(state, budget, deadline, once)
	local tried = { }
	-- Loss per point of removing each branch, from the last time it was evaluated. Removing a branch elsewhere
	-- in the tree rarely changes another branch's value much, so stale losses are good enough for ordering.
	local branchLoss = { }
	local function removeBranch(branch)
		local trial = self:CopyState(state)
		for _, id in ipairs(branch.ids) do
			self:RemoveNode(trial, id)
		end
		return trial
	end
	local improved = true
	local passes = 0
	while improved and not self.stopped and self:Now() < deadline and not (once and passes > 0) do
		passes = passes + 1
		improved = false
		local base = self:Evaluate(state)
		local branches = self:RemovableBranches(state, self.maxBranchSize)
		for _, branch in ipairs(branches) do
			branch.key = branchKey(state, branch)
			if not branchLoss[branch.key] and not self.stopped then
				branchLoss[branch.key] = (base.score - self:Evaluate(removeBranch(branch)).score) / branch.size
			end
			branch.loss = branchLoss[branch.key] or 0
		end
		t_sort(branches, function(a, b) return a.loss < b.loss end)

		for _, branch in ipairs(branches) do
			if self.stopped or self:Now() >= deadline then
				break
			end
			local key = branch.key
			if not tried[key] then
				tried[key] = true
				local banned = { }
				for _, id in ipairs(branch.ids) do
					banned[id] = true
				end
				local trial = removeBranch(branch)
				branchLoss[key] = (base.score - self:Evaluate(trial).score) / branch.size
				self:Fill(trial, budget, banned)
				-- Unban and top up in case points are left over
				self:Fill(trial, budget)
				if self:IsBetter(self:Evaluate(trial), base) then
					state = trial
					self:Track(state)
					tried = { }
					improved = true
					break
				end
			end
		end
		-- No single branch swap helps: try moves that change several places at once
		if not improved and not self.stopped and self:Now() < deadline then
			local trial = self:Reroute(state)
			if not trial and not self.stopped then
				trial = self:FarSwap(state, budget, self:BranchLossFunction(branchLoss))
			end
			if trial then
				state = trial
				self:Track(state)
				tried = { }
				improved = true
			end
		end
	end
	return state
end

-- Swaps a node that other nodes hang off (a travel node on a path) for an unallocated node that keeps everything
-- connected, e.g. taking the attack speed node instead of the Dexterity node on the way to a notable. Removing such a
-- node cuts off its branch, so Refine never tries this. Returns the best improving tree, or nil.
function TreeOptimizer:Reroute(state)
	local base = self:Evaluate(state)
	local bestState, bestEval
	local ids = { }
	for id in pairs(state.nodes) do
		t_insert(ids, id)
	end
	t_sort(ids)
	for _, id in ipairs(ids) do
		if self.stopped then
			break
		end
		local node = state.nodes[id]
		if not self.mustHave[id] and node.type ~= "Mastery" then
			local reached = self:Reachable(state.nodes, id)
			local cutOff = { }
			for otherId in pairs(state.nodes) do
				if otherId ~= id and not reached[otherId] then
					t_insert(cutOff, otherId)
				end
			end
			if cutOff[1] then
				-- A replacement has to touch the cut off part and the rest of the tree
				local touchesCut = { }
				for _, otherId in ipairs(cutOff) do
					for _, other in ipairs(self.spec.nodes[otherId].linked) do
						touchesCut[other.id] = true
					end
				end
				local set = copyTable(state.nodes, true)
				set[id] = nil
				for otherId in pairs(touchesCut) do
					local other = self.spec.nodes[otherId]
					if not state.nodes[otherId] and otherId ~= id and other.type ~= "Mastery" and self:IsEligible(other) then
						set[otherId] = other
						local reachedWith = self:Reachable(set)
						local connected = true
						for _, cutId in ipairs(cutOff) do
							if not reachedWith[cutId] then
								connected = false
								break
							end
						end
						set[otherId] = nil
						if connected and reachedWith[otherId] then
							local trial = self:CopyState(state)
							self:RemoveNode(trial, id)
							self:AddNode(trial, other)
							local eval = self:Evaluate(trial)
							if self:IsBetter(eval, bestEval or base) then
								bestState, bestEval = trial, eval
							end
						end
					end
				end
			end
		end
	end
	return bestState
end

-- Returns lossOf(from, branch) for FarSwap: score lost per point by removing `branch` from `from`, cached in `cache`
-- by the branch's nodes. Losses barely depend on the rest of the tree, so cached values are reused across trees.
function TreeOptimizer:BranchLossFunction(cache)
	return function(from, branch)
		local key = branchKey(from, branch)
		if not cache[key] then
			local without = self:CopyState(from)
			for _, id in ipairs(branch.ids) do
				self:RemoveNode(without, id)
			end
			cache[key] = (self:Evaluate(from).score - self:Evaluate(without).score) / branch.size
		end
		return cache[key]
	end
end

-- Far targets: notables, keystones and sockets further away than a fill step reaches
local FAR_REACH = 12
-- Far targets tried per call, best estimated gain per point first
local FAR_TRIES = 6

-- Takes the path to a far target and pays for it by dropping the cheapest branches anywhere in the tree, then tops
-- up any points left over. A person does this when they see that a keystone 6 points away beats a handful of weak
-- small nodes; the greedy fill can't, since something within reach always helps a little and Refine only frees one
-- branch at a time. `lossOf(trial, branch)` returns the score lost per point by removing a branch.
-- Returns the best improving tree, or nil.
function TreeOptimizer:FarSwap(state, budget, lossOf)
	local base = self:Evaluate(state)
	local baseKey = stateKey(state)
	self.farEstimates = self.farEstimates or { }
	self.farTried = self.farTried or { }
	self.farFailures = self.farFailures or { }
	local dist, parent = self:Frontier(state, FAR_REACH)
	local candidates = { }
	for id, d in pairs(dist) do
		local node = self.spec.nodes[id]
		if d > self.maxPathLength and node.type ~= "Normal" and node.type ~= "Mastery" and not self.farTried[baseKey .. ">" .. id] then
			local path = buildPath(node, dist, parent)
			-- Raw gain of adding the path on top of the tree, measured once and reused as an estimate afterwards
			if not self.farEstimates[id] then
				if self.stopped then
					return nil
				end
				local trial = self:CopyState(state)
				for i = #path, 1, -1 do
					self:AddNode(trial, path[i])
				end
				self.farEstimates[id] = self:Evaluate(trial).score - base.score
			end
			-- Targets whose swaps keep failing sink, so later calls try others
			t_insert(candidates, { id = id, path = path, priority = self.farEstimates[id] / d / (1 + (self.farFailures[id] or 0)) })
		end
	end
	t_sort(candidates, function(a, b)
		if a.priority ~= b.priority then
			return a.priority > b.priority
		end
		return a.id < b.id
	end)

	local bestState, bestEval
	for i = 1, m_min(FAR_TRIES, #candidates) do
		if self.stopped then
			break
		end
		local cand = candidates[i]
		if cand.priority <= 0 then
			break
		end
		self.farTried[baseKey .. ">" .. cand.id] = true
		local trial = self:CopyState(state)
		local onPath = { }
		for j = #cand.path, 1, -1 do
			self:AddNode(trial, cand.path[j])
			onPath[cand.path[j].id] = true
		end
		-- Drop the cheapest branches until back within budget
		while trial.count > budget and not self.stopped do
			local cheapest, cheapestLoss
			for _, branch in ipairs(self:RemovableBranches(trial, self.maxBranchSize)) do
				local touchesPath = false
				for _, id in ipairs(branch.ids) do
					if onPath[id] then
						touchesPath = true
						break
					end
				end
				if not touchesPath then
					local loss = lossOf(trial, branch)
					if not cheapestLoss or loss < cheapestLoss then
						cheapest, cheapestLoss = branch, loss
					end
				end
			end
			if not cheapest then
				break
			end
			for _, id in ipairs(cheapest.ids) do
				self:RemoveNode(trial, id)
			end
		end
		if trial.count <= budget then
			self:Fill(trial, budget)
			local eval = self:Evaluate(trial)
			if self:IsBetter(eval, bestEval or base) then
				bestState, bestEval = trial, eval
			else
				self.farFailures[cand.id] = (self.farFailures[cand.id] or 0) + 1
			end
		end
	end
	return bestState
end

-- Constraint repair
--
-- The fill and refine phases chase the stat, and the small deficit penalty in the score only steers them towards nodes
-- nearby. When the tree misses its requirements, Repair buys the path that adds the most merit per point (see Merit:
-- nearly always the one that removes the most deficit per point) from anywhere within REPAIR_REACH, e.g. reservation
-- efficiency on a cluster jewel across the tree, and pays for it with the branches that cost the least merit per point.
-- It repeats until the requirements are met or no path helps. Only nodes with a mod that can affect a violated
-- requirement are tried (see the `needs` of ConstraintDeficit), so a repair costs a few hundred calculations at most.
-- Greedy steps can take the long way round (several Strength nodes instead of the keystone that ignores attribute
-- requirements), so the best few paths that meet every requirement on their own are also tried, each as a single swap.

-- Furthest a repair reaches
local REPAIR_REACH = 12
-- Masteries further away than this are left out (each effect costs a calculation)
local REPAIR_MASTERY_REACH = 1
-- Paths in a row that fail to improve the tree before the greedy repair gives up
local REPAIR_TRIES = 3
-- Paths that meet every requirement on their own that are tried as single swaps
local REPAIR_FIXES = 2
-- Calculations one greedy step may spend on candidates whose value isn't known yet (nearest first); the rest wait for
-- the next step. A step that finds nothing within them still looks at every candidate.
local REPAIR_STEP_EVALS = 40

local function modsMatch(node, needs)
	for _, mod in ipairs(node.modList or { }) do
		for pattern in pairs(needs) do
			if mod.name:find(pattern, 1, true) then
				return true
			end
		end
	end
	return false
end

-- Whether `node`, `d` points from the tree, is worth trying in a repair. Returns the mastery effects to try for masteries.
function TreeOptimizer:RepairCandidate(state, node, d, needs)
	if node.type == "Mastery" then
		if d > REPAIR_MASTERY_REACH then
			return false
		end
		local effects = { }
		for _, effectId in ipairs(self:AvailableEffects(state, node)) do
			if modsMatch(self:GetEffectNode(node, effectId), needs) then
				t_insert(effects, effectId)
			end
		end
		return effects[1] ~= nil, effects
	end
	if node.type == "Keystone" or (node.isJewelSocket and self.spec.jewels[node.id]) then
		return true
	end
	return modsMatch(node, needs)
end

-- Adds `path` (target first, as buildPath returns it) to a copy of `state`, then removes the branches elsewhere that cost
-- the least merit per point until the tree is back within budget. Returns nil if it can't get back within budget.
function TreeOptimizer:BuyAndSell(state, path, effectId, budget, lossOf)
	local trial = self:CopyState(state)
	local onPath = { }
	for i = #path, 1, -1 do
		self:AddNode(trial, path[i], i == 1 and effectId or nil)
		onPath[path[i].id] = true
	end
	while trial.count > budget and not self.stopped do
		local cheapest, cheapestLoss
		for _, branch in ipairs(self:RemovableBranches(trial, self.maxBranchSize)) do
			local touchesPath = false
			for _, id in ipairs(branch.ids) do
				if onPath[id] then
					touchesPath = true
					break
				end
			end
			if not touchesPath then
				local loss = lossOf(trial, branch)
				if not cheapestLoss or loss < cheapestLoss then
					cheapest, cheapestLoss = branch, loss
				end
			end
		end
		if not cheapest then
			return nil
		end
		for _, id in ipairs(cheapest.ids) do
			self:RemoveNode(trial, id)
		end
	end
	return trial.count <= budget and trial or nil
end

function TreeOptimizer:Repair(state, budget)
	local base = self:Evaluate(state)
	if base.deficit == 0 then
		return state
	end
	local original, originalEval = state, base
	-- Merit lost per point by removing a branch, cached by its nodes (see BranchLossFunction)
	local losses = { }
	local function lossOf(from, branch)
		local key = branchKey(from, branch)
		if not losses[key] then
			local without = self:CopyState(from)
			for _, id in ipairs(branch.ids) do
				self:RemoveNode(without, id)
			end
			losses[key] = (self:Merit(self:Evaluate(from)) - self:Merit(self:Evaluate(without))) / branch.size
		end
		return losses[key]
	end
	-- Spends points a swap freed beyond what it needed. The fill never makes a tree that meets its requirements miss them.
	local function topUp(trial)
		if trial.count < budget then
			self:Fill(trial, budget)
		end
		return trial
	end
	-- [targetId] = { gain = merit gain per point, key = state key it was measured on, effectId }, and estimates for
	-- targets never evaluated, by signature (see NodeSignature)
	local estimates, signatureRates = { }, { }
	-- Paths from the original tree that meet every requirement on their own
	local fixes = { }
	local failures = 0
	while base.deficit > 0 and failures < REPAIR_TRIES and not self.stopped do
		local baseKey = stateKey(state)
		local dist, parent = self:Frontier(state, REPAIR_REACH)
		local candidates = { }
		for id, d in pairs(dist) do
			local node = self.spec.nodes[id]
			local ok, effects = self:RepairCandidate(state, node, d, base.needs)
			if ok then
				t_insert(candidates, { id = id, node = node, dist = d, effects = effects, signature = self:NodeSignature(node) })
			end
		end
		t_sort(candidates, function(a, b)
			if a.dist ~= b.dist then
				return a.dist < b.dist
			end
			return a.id < b.id
		end)
		-- Lazy greedy on merit gain per point, as in Fill: re-evaluate the best estimate until it is up to date
		local accepted
		local stepEvals, limit = 0, REPAIR_STEP_EVALS
		while not self.stopped do
			local best, bestPriority
			for _, cand in ipairs(candidates) do
				local est = estimates[cand.id]
				local priority = est and est.gain or signatureRates[cand.signature] or math.huge
				if stepEvals >= limit and priority == math.huge then
					priority = -math.huge
				end
				if not cand.done and (not best or priority > bestPriority) then
					best, bestPriority = cand, priority
				end
			end
			if not best or bestPriority <= EPSILON then
				-- Nothing helps among the candidates looked at: look at the rest
				if stepEvals >= limit and not accepted and not self.stopped then
					local unknown = false
					for _, cand in ipairs(candidates) do
						if not cand.done and not estimates[cand.id] and not signatureRates[cand.signature] then
							unknown = true
							break
						end
					end
					if unknown then
						limit = math.huge
						goto nextCandidate
					end
				end
				break
			end
			local est = estimates[best.id]
			if est and est.key == baseKey then
				accepted = best
				break
			end
			stepEvals = stepEvals + 1
			local path = self:TargetPath(state, best.node, dist, parent)
			local trial = self:CopyState(state)
			for i = #path, 2, -1 do
				self:AddNode(trial, path[i])
			end
			local bestEval, bestEffect
			for _, effectId in ipairs(best.effects or { false }) do
				local withTarget = self:CopyState(trial)
				self:AddNode(withTarget, best.node, effectId or nil)
				local eval = self:Evaluate(withTarget)
				if not bestEval or self:Merit(eval) > self:Merit(bestEval) then
					bestEval, bestEffect = eval, effectId
				end
			end
			-- Only paths that reduce the deficit repair anything
			local gain = bestEval.deficit < base.deficit - EPSILON and (self:Merit(bestEval) - self:Merit(base)) / #path or -math.huge
			estimates[best.id] = { gain = gain, key = baseKey, effectId = bestEffect or nil }
			signatureRates[best.signature] = gain
			best.path = path
			if bestEval.deficit == 0 and state == original then
				t_insert(fixes, { path = path, effectId = bestEffect or nil, gain = gain })
			end
			::nextCandidate::
		end
		if not accepted then
			break
		end
		accepted.done = true
		local trial = self:BuyAndSell(state, accepted.path or self:TargetPath(state, accepted.node, dist, parent), estimates[accepted.id].effectId, budget, lossOf)
		local eval = trial and self:Evaluate(trial)
		if trial and self:IsBetter(eval, base) then
			state, base = trial, eval
			failures = 0
		else
			failures = failures + 1
		end
	end
	state = topUp(state)

	-- Single swaps for the paths that meet every requirement on their own
	t_sort(fixes, function(a, b) return a.gain > b.gain end)
	for i = 1, m_min(REPAIR_FIXES, #fixes) do
		if self.stopped then
			break
		end
		local trial = self:BuyAndSell(original, fixes[i].path, fixes[i].effectId, budget, lossOf)
		if trial then
			trial = topUp(trial)
			if self:IsBetter(self:Evaluate(trial), self:Evaluate(state)) then
				state = trial
			end
		end
	end
	return self:IsBetter(self:Evaluate(state), originalEval) and state or original
end

-- Node value model
--
-- A full calculation takes 10-25 ms, which limits the search to tens of thousands of trees. The model measures what
-- each node near a base tree is worth on its own (one calculation each: the calculator doesn't need nodes to be
-- connected) and predicts any tree as the base plus the summed effects of the nodes it adds, minus those of the nodes
-- it drops, on the log of the stat and on every value the constraints read. Predictions take microseconds, so the
-- usual moves can search the model exhaustively; the trees it picks are then checked with full calculations. Measured
-- on real builds, predictions stay within ~1% for trees up to ~20 changed nodes and degrade gradually further out,
-- mostly where stats interact (crit, conversion, thresholds).
--
-- The model is built incrementally and kept for the whole run. Each round measures one more chunk, nearest and most
-- useful first (dropping allocated nodes, then unallocated nodes one step out, then further), then searches what is
-- known and checks its best trees, so the first improvements come after a few hundred calculations instead of
-- after measuring everything. When the best tree changes, the measured effects carry over as estimates (a node's
-- effect rarely changes much when other parts of the tree change) and are re-measured on the new tree over the
-- following rounds.

-- How far from the base tree unallocated nodes are measured, and unallocated masteries (every effect costs a
-- calculation, and a mastery further out rarely makes the difference)
local MODEL_RADIUS = 6
local MODEL_MASTERY_RADIUS = 2
-- Distinct trees found on the model that are checked with full calculations per round
local MODEL_CHECKS = 6
-- Checked trees that miss a requirement and beat the base's stat that get a repair, per round
local MODEL_REPAIRS = 2
-- Seconds the search on the model may take per round: a share of the time the round spent measuring, within limits.
-- It stops earlier once it has gone MODEL_SEARCH_STALL seconds (or as long as it took to find its best tree, if
-- longer) without finding a better tree.
local MODEL_SEARCH_TIME = 5
local MODEL_SEARCH_MIN = 0.5
local MODEL_SEARCH_SHARE = 0.25
local MODEL_SEARCH_STALL = 0.5
-- Calculations the first round spends measuring (it then finishes the distance band it is in), and how much more
-- each following round may spend
local MODEL_CHUNK_MIN = 150
local MODEL_CHUNK_GROWTH = 8
-- Measurements made on an earlier base tree are redone only after everything within MODEL_RADIUS has been measured
-- (they're queued this many distance bands further out): measuring further out finds far more than refreshing
local MODEL_STALE_DELAY = 100
-- Model rounds rest after this many fail in a row, or one fails with nothing left to measure, until the best tree changes
local MODEL_MAX_FAILURES = 3
-- Share of the run the model may use (measuring, searching and checking), and the longest one round may spend measuring,
-- as a share of the time limit. The rest goes to the real search, which keeps improving long after the model has
-- measured what is worth measuring.
local MODEL_SHARE = 0.4
local MODEL_MEASURE_SHARE = 0.08

-- Effect of moving from `base` to `eval` on the modelled values
-- Second result: for the discovered paths, the change of their log where they are positive on both sides; third result:
-- their raw change where not (only those paths). Predictions use the log changes while a path is positive on the base.
local function fieldDelta(base, fields, paths)
	local delta = { }
	local dl, dr
	if paths and #paths > 0 then
		dl, dr = { }, { }
		for _, path in ipairs(paths) do
			local a, b = base[path], fields[path]
			if a and b and a > 0 and b > 0 then
				dl[path] = m_log(b / a)
			elseif a or b then
				dr[path] = (b or 0) - (a or 0)
			end
		end
	end
	for key, value in pairs(fields) do
		if type(value) == "number" then
			delta[key] = value - (base[key] or 0)
		end
	end
	for key, value in pairs(base) do
		if type(value) == "number" and fields[key] == nil then
			delta[key] = -value
		end
	end
	return delta, dl, dr
end

-- Number of nodes that differ between two trees (a mastery effect switch counts as one)
local function changedNodes(a, b)
	local count = 0
	for id in pairs(a.nodes) do
		if not b.nodes[id] or a.mastery[id] ~= b.mastery[id] then
			count = count + 1
		end
	end
	for id in pairs(b.nodes) do
		if not a.nodes[id] then
			count = count + 1
		end
	end
	return count
end

-- Model entries are keyed like stateKey: the node id, or "id:effect" for a mastery with an effect
local function nodeKey(id, effectId)
	return effectId and (id .. ":" .. effectId) or id
end

-- The model: `with[key]` = { delta = effect of having the node (with that mastery effect) rather than not, version =
-- base it was measured on }; `known[id]` marks nodes with at least one entry; `suspect[key]` marks stale entries that
-- a failed check used, to be re-measured first
function TreeOptimizer:NewModel()
	return { version = 0, with = { }, known = { }, suspect = { }, bases = { }, drift = { } }
end

-- Moves the model's base to `base`, keeping every measurement as an estimate
function TreeOptimizer:RebaseModel(model, base)
	model.version = model.version + 1
	model.base = self:CopyState(base)
	model.baseKey = stateKey(base)
	model.baseFields = self:Evaluate(model.base).fields
	-- Paths whose changes are predicted in log space: those positive on the base
	model.logMode = { }
	for _, path in ipairs(self.discPaths) do
		local v = model.baseFields[path]
		model.logMode[path] = v ~= nil and v > 0
	end
	model.bases[model.version] = model.base
	model.drift = { [model.version] = 0 }
	model.shared = { }
	model.searched = false
end

-- Nodes changed between the base an entry was measured on and the current base
function TreeOptimizer:ModelDrift(model, version)
	if not model.drift[version] then
		model.drift[version] = changedNodes(model.bases[version], model.base)
	end
	return model.drift[version]
end

-- Measurements still wanted around the model's base, in the order to make them: suspects, then by distance band
-- (0: dropping each allocated node, 1: other effects of allocated masteries and unallocated nodes next to the tree, and
-- so on), with unmeasured entries before stale ones a few bands nearer
function TreeOptimizer:ModelTasks(model)
	local base = model.base
	local tasks = { }
	local function want(id, effectId, band)
		local key = nodeKey(id, effectId)
		local entry = model.with[key]
		local priority = band
		if entry then
			if self:ModelDrift(model, entry.version) == 0 then
				return
			end
			priority = model.suspect[key] and -1 or (band + MODEL_STALE_DELAY)
		end
		t_insert(tasks, { key = key, id = id, effectId = effectId, priority = priority })
	end
	-- Must-haves are never dropped or switched by the search
	for id, node in pairs(base.nodes) do
		if not self.mustHave[id] then
			want(id, base.mastery[id], 0)
			if node.type == "Mastery" then
				for _, effectId in ipairs(self:AvailableEffects(base, node)) do
					if effectId ~= base.mastery[id] then
						want(id, effectId, 1)
					end
				end
			end
		end
	end
	local dist = self:Frontier(base, MODEL_RADIUS)
	for id, d in pairs(dist) do
		local node = self.spec.nodes[id]
		if node.type == "Mastery" then
			if d <= MODEL_MASTERY_RADIUS then
				for _, effectId in ipairs(self:AvailableEffects(base, node)) do
					want(id, effectId, d + 1)
				end
			end
		else
			want(id, nil, d)
		end
	end
	t_sort(tasks, function(a, b)
		if a.priority ~= b.priority then
			return a.priority < b.priority
		end
		return tostring(a.key) < tostring(b.key)
	end)
	return tasks
end

-- Nodes whose effect may depend on where they are: those in the radius of a socketed jewel (radius jewels change or
-- count the nodes around them) or conquered by a timeless jewel, and jewel sockets
function TreeOptimizer:PositionDependent(id)
	if not self.positionDependent then
		local set = copyTable(self.conqueredBySocket, true)
		for socketId, itemId in pairs(self.spec.jewels) do
			local socket = self.spec.nodes[socketId]
			local item = self.build.itemsTab.items[itemId]
			if socket and item and item.jewelRadiusIndex and socket.nodesInRadius then
				for nodeId in pairs(socket.nodesInRadius[item.jewelRadiusIndex] or { }) do
					set[nodeId] = true
				end
			end
		end
		self.positionDependent = set
	end
	local node = self.spec.nodes[id]
	return self.positionDependent[id] or node.type == "Socket" or node.isJewelSocket or node.expansionJewel ~= nil
end

-- Nodes with the same stats have the same effect wherever they are (unless PositionDependent), so on the same base
-- tree one measurement serves all of them: travel nodes, cluster small nodes and masteries of the same kind repeat
-- all over the tree. Returns the key such nodes share, or nil. Allocated and unallocated nodes don't share (dropping
-- one of several copies isn't worth the same as adding another), nor do effect switches of allocated masteries
-- (each is measured without that mastery's current effect).
function TreeOptimizer:ModelSignature(model, task)
	local id, effectId = task.id, task.effectId
	if self:PositionDependent(id) then
		return nil
	end
	local side = model.base.nodes[id] and "in" or "out"
	if effectId then
		if model.base.nodes[id] and effectId ~= model.base.mastery[id] then
			return nil
		end
		return side .. ":mastery:" .. effectId
	end
	local node = self.spec.nodes[id]
	return side .. ":" .. tostring(node.type) .. ":" .. t_concat(node.sd or { }, "\n")
end

-- Measures one entry on the model's base: the base with and without the node. One side is the base itself, except
-- when switching an allocated mastery, whose "without" side is shared by all its effects.
function TreeOptimizer:MeasureModelTask(model, task)
	local base = model.base
	local id, effectId = task.id, task.effectId
	local signature = self:ModelSignature(model, task)
	local shared = signature and model.shared[signature]
	if shared then
		model.with[task.key] = { delta = shared[1], dl = shared[2], dr = shared[3], version = model.version }
		model.known[id] = true
		model.suspect[task.key] = nil
		model.searched = false
		return
	end
	local withState, withoutState
	if base.nodes[id] then
		withoutState = self:CopyState(base)
		self:RemoveNode(withoutState, id)
		if effectId == base.mastery[id] then
			withState = base
		else
			withState = self:CopyState(withoutState)
			self:AddNode(withState, self.spec.nodes[id], effectId)
		end
	else
		withoutState = base
		withState = self:CopyState(base)
		self:AddNode(withState, self.spec.nodes[id], effectId)
	end
	local withFields = self:Evaluate(withState).fields
	local withoutFields = self:Evaluate(withoutState).fields
	local delta, dl, dr = fieldDelta(withoutFields, withFields, self.structured and self.discPaths)
	if signature then
		model.shared[signature] = { delta, dl, dr }
	end
	model.with[task.key] = { delta = delta, dl = dl, dr = dr, version = model.version }
	model.known[id] = true
	model.suspect[task.key] = nil
	model.searched = false
end

-- Measures the whole model around `base` (or until `deadline`). Returns nil if stopped part way.
function TreeOptimizer:BuildModel(base, deadline)
	local model = self:NewModel()
	self:RebaseModel(model, base)
	for _, task in ipairs(self:ModelTasks(model)) do
		if self.stopped or (deadline and self:Now() >= deadline) then
			return nil
		end
		self:MeasureModelTask(model, task)
	end
	return model
end

-- Predicted evaluation of `state`, or nil if it uses a node the model didn't measure
function TreeOptimizer:ModelEvaluate(model, state)
	local fields, changed = self:ModelFields(model, state)
	if not fields then
		return nil
	end
	local deficit, attributes, violations = self:ConstraintDeficit(fields)
	local penalty = self.distancePenalty and self:DistancePenalty(changed) or 0
	local deficitCost = deficit * ATTRIBUTE_PENALTY
	return {
		value = math.exp(fields.logValue - penalty),
		deficit = deficit,
		attributes = attributes,
		violations = violations,
		score = fields.logValue - penalty - deficitCost,
		-- What the model really predicts, before the pessimism for the distance (the prediction log and RecordCheck)
		rawScore = fields.logValue - deficitCost,
		plainScore = (fields.plainLogValue or fields.logValue) - deficitCost,
		structScore = fields.structLogValue and (fields.structLogValue - deficitCost),
		changed = changed,
		predPaths = fields.predPaths,
	}
end

-- Pessimism, in log score, for a predicted tree that differs from the model's base in `changed` nodes. The model's
-- predictions are optimistic (the best of many predictions is picked) and the more so the further the tree is from the
-- base, so the checks of its predictions decide: RecordCheck keeps the mean (predicted - real) per distance band, and
-- this interpolates it. Nothing is subtracted before the first checks.
local DISTANCE_BANDS = { 5, 10, 20 } -- upper ends of the bands; the last one is open
local BAND_CENTRES = { 3, 8, 15, 30 }
local function distanceBand(changed)
	for i, upper in ipairs(DISTANCE_BANDS) do
		if changed <= upper then
			return i
		end
	end
	return #DISTANCE_BANDS + 1
end

function TreeOptimizer:DistancePenalty(changed)
	local curve = self.calib.curve
	local n = #curve
	if n == 1 then
		return 0
	end
	for i = 2, n do
		local d0, b0 = curve[i - 1][1], curve[i - 1][2]
		local d1, b1 = curve[i][1], curve[i][2]
		if changed <= d1 then
			return b0 + (b1 - b0) * (changed - d0) / (d1 - d0)
		end
	end
	-- Beyond the last band measured: error grows faster than linearly with distance
	local d1, b1 = curve[n][1], curve[n][2]
	local slope = 2 * b1 / d1
	if n > 2 then
		slope = m_max(slope, (b1 - curve[n - 1][2]) / (d1 - curve[n - 1][1]))
	end
	return b1 + slope * (changed - d1)
end

-- Whether to predict with the structured model: until its checks show it less accurate than the sum of log deltas
function TreeOptimizer:StructuredWins()
	local calib = self.calib
	return calib.n < 3 or calib.structErr <= calib.plainErr * 1.05
end

-- Learns from a model prediction (`predicted`, a ModelEvaluate result) and the calculation that checked it
function TreeOptimizer:RecordCheck(predicted, real)
	if not (predicted.rawScore and real.score > -math.huge and predicted.deficit == 0 and real.deficit == 0) then
		return
	end
	local calib = self.calib
	local bucket = calib.buckets[distanceBand(predicted.changed)]
	bucket.n = bucket.n + 1
	bucket.sum = bucket.sum + (predicted.rawScore - real.score)
	if predicted.structScore then
		calib.n = calib.n + 1
		calib.plainErr = calib.plainErr + math.abs(predicted.plainScore - real.score)
		calib.structErr = calib.structErr + math.abs(predicted.structScore - real.score)
	end
	-- Bands with a few checks give the bias (shrunk towards 0 while they are few); it never falls with distance
	local curve, previous = { { 0, 0 } }, 0
	for i, b in ipairs(calib.buckets) do
		if b.n >= 2 then
			previous = m_max(previous, b.sum / b.n * b.n / (b.n + 3))
			t_insert(curve, { BAND_CENTRES[i], previous })
		end
	end
	calib.curve = curve
end

-- Predicted values (log of the stat and everything the constraints read) of `state`, relative to the measured base
function TreeOptimizer:ModelFields(model, state)
	local base = model.base
	local with = model.with
	local fields = copyTable(model.baseFields, true)
	local structured = self.structured and #self.discPaths > 0
	local logSum, rawSum, logMode = { }, { }, model.logMode
	local changed = 0
	local function apply(key, sign)
		local entry = with[key]
		if not entry then
			return false
		end
		for field, value in pairs(entry.delta) do
			fields[field] = (fields[field] or 0) + sign * value
		end
		if structured and entry.dl then
			for path, value in pairs(entry.dl) do
				if logMode[path] then
					logSum[path] = (logSum[path] or 0) + sign * value
				end
			end
			for path, value in pairs(entry.dr) do
				if logMode[path] then
					rawSum[path] = (rawSum[path] or 0) + sign * value
				end
			end
		end
		return true
	end
	for id in pairs(state.nodes) do
		local effectId = state.mastery[id]
		if not base.nodes[id] then
			changed = changed + 1
			if not apply(nodeKey(id, effectId), 1) then
				return nil
			end
		elseif effectId ~= base.mastery[id] then
			changed = changed + 1
			if not apply(nodeKey(id, effectId), 1) or not apply(nodeKey(id, base.mastery[id]), -1) then
				return nil
			end
		end
	end
	for id in pairs(base.nodes) do
		if not state.nodes[id] then
			changed = changed + 1
			if not apply(nodeKey(id, base.mastery[id]), -1) then
				return nil
			end
		end
	end
	if structured then
		local predicted = self:PredictLogValue(model, fields, logSum, rawSum)
		if predicted then
			fields.plainLogValue, fields.structLogValue = fields.logValue, predicted
			if self:StructuredWins() then
				fields.logValue = predicted
			end
		end
	end
	return fields, changed
end

-- Log of the score function on the output predicted for `fields`: the base's value of each discovered path moved by
-- the summed log changes (and the raw changes of nodes that crossed zero) where the path is positive on the base, by
-- the summed raw changes otherwise. Nil if the function can't run on it (the caller keeps the summed log of the stat).
function TreeOptimizer:PredictLogValue(model, fields, logSum, rawSum)
	local baseFields, logMode = model.baseFields, model.logMode
	local predicted = setmetatable({ }, { __index = self.template })
	local values = self.predictionLog and { } or nil
	fields.predPaths = values
	for _, path in ipairs(self.discPaths) do
		local value
		local base = baseFields[path]
		if logMode[path] then
			value = base * math.exp(logSum[path] or 0) + (rawSum[path] or 0)
		elseif base ~= nil or fields[path] ~= 0 then
			value = fields[path]
		else
			value = false -- absent on the base and unchanged: hide the template's value
		end
		local parts = self.discParts[path]
		local target = predicted
		local source = self.template
		for i = 1, #parts - 1 do
			local part = parts[i]
			source = type(source) == "table" and source[part] or nil
			local nested = rawget(target, part)
			if not nested then
				nested = setmetatable({ }, { __index = type(source) == "table" and source or nil })
				target[part] = nested
			end
			target = nested
		end
		target[parts[#parts]] = value
		if values then
			values[path] = value or nil
		end
	end
	local ok, result = pcall(self.scoreFunc, predicted)
	if not ok or type(result) ~= "number" or result ~= result then
		return nil
	end
	return result > 0 and m_log(result) or (-100 + result * EPSILON)
end

-- Keys of the entries ModelFields uses to predict `state`
local function changedKeys(base, state)
	local keys = { }
	for id in pairs(state.nodes) do
		local effectId = state.mastery[id]
		if not base.nodes[id] or effectId ~= base.mastery[id] then
			t_insert(keys, nodeKey(id, effectId))
			if base.nodes[id] then
				t_insert(keys, nodeKey(id, base.mastery[id]))
			end
		end
	end
	for id in pairs(base.nodes) do
		if not state.nodes[id] then
			t_insert(keys, nodeKey(id, base.mastery[id]))
		end
	end
	return keys
end

-- One model round around the best tree: measure the next chunk of the model, search what it knows with the usual
-- moves, then check the most promising trees it found with full calculations. Returns the best of those if it really
-- beats the base, else nil, and whether measurements are still pending around the base.
function TreeOptimizer:ModelRound(deadline)
	local model = self.model
	if not model then
		model = self:NewModel()
		self.model = model
	end
	if model.baseKey ~= stateKey(self.best) then
		self:RebaseModel(model, self.best)
	end
	local base = model.base
	local baseEval = self:Evaluate(base)

	-- Measure the next chunk, finishing the distance band it is in once past the chunk size. Chunks grow
	-- geometrically: the first gives a quick first search, the next ones cover the rest of the model.
	self.phase = "node values"
	local tasks = self:ModelTasks(model)
	local startEvals = self.evalCount
	local measureStart = self:Now()
	local chunk = MODEL_CHUNK_MIN * MODEL_CHUNK_GROWTH ^ (model.chunks or 0)
	local done = 0
	for i, task in ipairs(tasks) do
		local spent = self.evalCount - startEvals
		if self.stopped or self:Now() >= deadline
			or (spent >= chunk and math.floor(task.priority) ~= math.floor(tasks[i - 1].priority))
			or (i > 1 and self:Now() - measureStart >= self.timeLimit * (self.modelMeasureShare or MODEL_MEASURE_SHARE)) then
			break
		end
		self:MeasureModelTask(model, task)
		done = i
	end
	if self.evalCount > startEvals then
		model.chunks = (model.chunks or 0) + 1
	end
	local measureTime = self:Now() - measureStart
	local pending = #tasks > done
	if self.stopped or self:Now() >= deadline or model.searched then
		return nil, pending
	end
	model.searched = true

	-- A copy of the optimizer that scores trees with the model. It shares the graph and settings, but has its own
	-- caches and bookkeeping so nothing predicted leaks into the real search.
	self.phase = "model search"
	local searchStart = self:Now()
	local searchTime = m_max(MODEL_SEARCH_MIN, MODEL_SEARCH_SHARE * measureTime)
	local searchDeadline = m_min(deadline, searchStart + m_min(searchTime, MODEL_SEARCH_TIME))
	local startTime, lastFound, startDeadline = searchStart, searchStart, searchDeadline
	local unknown = { score = -math.huge, deficit = math.huge, value = 0, attributes = { }, violations = { } }
	local proxy = setmetatable({
		scoreCache = { }, estimates = { }, farEstimates = { }, farTried = { }, farFailures = { }, stopped = false,
	}, { __index = self })
	local predictions = 0
	proxy.Evaluate = function(p, state)
		if not state.eval then
			state.eval = self:ModelEvaluate(model, state) or unknown
			predictions = predictions + 1
			if self.evalClock then
				self.virtualClock.extra = self.virtualClock.extra + PREDICTION_COST
			end
			-- Predictions don't go through the real Evaluate, so report progress (letting the UI run) and honour Stop here.
			-- The search also ends once it stops finding better trees.
			if predictions % 50 == 0 then
				if self.onProgress then
					self.onProgress({ phase = self.phase, evals = self.evalCount, best = self.best and self.best.eval, elapsed = self:Now() - self.startTime, timeLimit = self.timeLimit })
				end
				if self.shouldStop and self.shouldStop() then
					self.stopped = true
					p.stopped = true
				end
				local t = self:Now()
				if t >= startDeadline or t - lastFound > m_max(MODEL_SEARCH_STALL, lastFound - startTime) then
					p.stopped = true
				end
			end
		end
		return state.eval
	end
	proxy.IsEligible = function(p, node)
		if not self:IsEligible(node) then
			return false
		end
		return base.nodes[node.id] ~= nil or model.known[node.id] == true
	end
	-- Candidates for checking: the best predicted tree within each distance of the base (predictions are most
	-- reliable near the base), and the best distinct trees overall. Trees already calculated count with their real
	-- score, so the search doesn't keep returning trees that failed their check.
	local bestWithin = { }
	local limits = { 5, 10, 20, math.huge }
	local top = { }
	local function better(a, b)
		return self:IsBetter(a.eval, b.eval)
	end
	proxy.Track = function(p, state)
		local eval = p:Evaluate(state)
		if state.count > self.points or not self:IsBetter(eval, baseEval) then
			return
		end
		local key = stateKey(state)
		local real = self.scoreCache[key]
		if real then
			if not self:IsBetter(real, baseEval) then
				return
			end
			eval = real
		end
		local changed = changedNodes(state, base)
		local entry
		for i, limit in ipairs(limits) do
			if changed <= limit and (not bestWithin[i] or self:IsBetter(eval, bestWithin[i].eval)) then
				entry = entry or { state = p:CopyState(state), eval = eval, key = key }
				bestWithin[i] = entry
				entry.predicted = not real
				lastFound = self:Now()
			end
		end
		if #top < MODEL_CHECKS or self:IsBetter(eval, top[#top].eval) then
			for _, other in ipairs(top) do
				if other.key == key then
					return
				end
			end
			t_insert(top, entry or { state = p:CopyState(state), eval = eval, key = key, predicted = not real })
			t_sort(top, better)
			top[MODEL_CHECKS + 1] = nil
			lastFound = self:Now()
		end
	end

	-- Search the model: refine from a start tree, then perturb and refine (predictions are cheap, so this explores far
	-- more than the real search can). The search starts from the base, then from the tree the run started from: the
	-- first rounds only know the nodes next to the tree, and the small improvements they find can lead into a worse
	-- basin than the one a search of the whole model finds from the start tree.
	local starts = { base }
	if self.startState and stateKey(self.startState) ~= model.baseKey then
		t_insert(starts, self.startState)
	end
	for i, start in ipairs(starts) do
		if self.stopped then
			break
		end
		proxy.stopped = false
		startTime = self:Now()
		lastFound = startTime
		startDeadline = m_min(searchDeadline, startTime + (searchDeadline - startTime) / (#starts - i + 1))
		local trial = proxy:CopyState(start)
		trial.eval = nil
		proxy:Fill(trial, self.points)
		local current = proxy:Refine(trial, self.points, startDeadline)
		local kicks, kickFailures = 0, 0
		while not proxy.stopped and self:Now() < startDeadline do
			kicks = kicks + 1
			local perturbed
			if kicks % 3 == 0 then
				perturbed = proxy:PerturbRegion(current, self.points, 2 + m_floor(kicks / 3) % 4)
			end
			perturbed = perturbed or proxy:Perturb(current, self.points, m_min(2 + m_floor(kickFailures / 4), 8))
			perturbed = proxy:Refine(perturbed, self.points, startDeadline)
			if self:IsBetter(proxy:Evaluate(perturbed), proxy:Evaluate(current)) then
				current = perturbed
				kickFailures = 0
			else
				kickFailures = kickFailures + 1
			end
		end
	end

	-- Check candidates with full calculations, nearest to the base first. Stale entries behind a failed prediction
	-- are re-measured first in the next round.
	self.phase = "model check"
	local candidates = { }
	for i = 1, #limits do
		if bestWithin[i] then
			t_insert(candidates, bestWithin[i])
		end
	end
	for _, entry in ipairs(top) do
		t_insert(candidates, entry)
	end
	local bestState, bestEval
	local checked, checks, repairs = { }, 0, 0
	for _, candidate in ipairs(candidates) do
		if self.stopped or checks >= MODEL_CHECKS then
			break
		end
		if not checked[candidate.key] then
			checked[candidate.key] = true
			checks = checks + 1
			local state = self:CopyState(candidate.state)
			-- The model's own tree, for judging its entries: a repair swaps in nodes it knows nothing about
			local modelTree = self:CopyState(state)
			state.eval = nil
			local eval = self:Evaluate(state)
			self:Track(state)
			if candidate.predicted then
				self:RecordCheck(candidate.eval, eval)
			end
			if self.predictionLog and candidate.predicted and candidate.eval.score > -math.huge then
				local predicted = candidate.eval
				t_insert(self.predictionLog, {
					predScore = predicted.rawScore or predicted.score, realScore = eval.score,
					structScore = predicted.structScore,
					predDeficit = predicted.deficit, realDeficit = eval.deficit,
					baseScore = baseEval.score, baseDeficit = baseEval.deficit,
					predBetter = self:IsBetter({ score = predicted.rawScore or predicted.score, deficit = predicted.deficit }, baseEval), realBetter = self:IsBetter(eval, baseEval),
					changed = changedNodes(candidate.state, base), evals = self.evalCount,
					predPaths = predicted.predPaths, plainScore = predicted.plainScore, realPaths = self:DiscoveredValues(eval.fields),
				})
			end
			-- A tree that is stronger but misses a requirement by a little (the model reads the requirements less
			-- accurately than the stat) gets a repair
			if eval.deficit > 0 and baseEval.deficit == 0 and eval.value > baseEval.value and repairs < MODEL_REPAIRS then
				repairs = repairs + 1
				state = self:Repair(state, self.points)
				eval = self:Evaluate(state)
				self:Track(state)
			end
			if self:IsBetter(eval, bestEval or baseEval) then
				bestState, bestEval = state, eval
			elseif not self:IsBetter(eval, baseEval) then
				for _, key in ipairs(changedKeys(base, modelTree)) do
					local entry = model.with[key]
					if entry and entry.version ~= model.version then
						model.suspect[key] = true
						pending = true
					end
				end
			end
		end
	end
	return bestState, pending
end

-- Removes a few random branches from `state` and refills without them, to escape a local optimum
function TreeOptimizer:Perturb(state, budget, strength)
	local trial = self:CopyState(state)
	local banned = { }
	strength = strength or 2
	local count = self.random(strength, strength + 2)
	for _ = 1, count do
		local branches = self:RemovableBranches(trial, self.maxBranchSize)
		if not branches[1] then
			break
		end
		local branch = branches[self.random(1, #branches)]
		for _, id in ipairs(branch.ids) do
			self:RemoveNode(trial, id)
			banned[id] = true
		end
	end
	self:Fill(trial, budget, banned)
	self:Fill(trial, budget)
	return trial
end

-- Removes every allocated node within `radius` links of a random allocated notable (a whole region of the tree, which
-- Perturb's small branches can't do), then refills without those nodes. Returns nil if the must-haves can't be kept.
function TreeOptimizer:PerturbRegion(state, budget, radius)
	local trial = self:CopyState(state)
	local centres = { }
	for id, node in pairs(trial.nodes) do
		if (node.type == "Notable" or node.type == "Keystone") and not self.mustHave[id] then
			t_insert(centres, id)
		end
	end
	if not centres[1] then
		return nil
	end
	t_sort(centres)
	local centre = self.spec.nodes[centres[self.random(1, #centres)]]
	local dist = { [centre.id] = 0 }
	local queue = { centre }
	local head = 1
	local banned = { }
	while queue[head] do
		local node = queue[head]
		head = head + 1
		local id = node.id
		if trial.nodes[id] and not self.mustHave[id] then
			self:RemoveNode(trial, id)
			banned[id] = true
		end
		if dist[id] < radius then
			for _, other in ipairs(node.linked) do
				if not dist[other.id] and other ~= self.root and not other.ascendancyName and other.type ~= "ClassStart" and other.type ~= "AscendClassStart" then
					dist[other.id] = dist[id] + 1
					t_insert(queue, other)
				end
			end
		end
	end
	self:Prune(trial)
	if not self:ConnectMustHaves(trial) or trial.count > budget then
		return nil
	end
	self:Fill(trial, budget, banned)
	self:Fill(trial, budget)
	return trial
end

-- Main entry point. Returns a result table; call :ApplyToNewSpec(result) to add it to the build.
function TreeOptimizer:Run()
	local startTime = self:Now()
	self.startTime = startTime
	local state = self:NewState()

	-- Phase 1: seed
	self.phase = "seed"
	local initial = self:NewState()
	for id, node in pairs(self.spec.allocNodes) do
		if not node.ascendancyName and node.type ~= "ClassStart" and node.type ~= "AscendClassStart" then
			self:AddNode(initial, node)
		end
	end
	local initialEval = self:Evaluate(initial)
	self.initialEval = initialEval
	if not self.fromScratch then
		state = self:CopyState(initial)
		for id in pairs(self.mustNot) do
			self:RemoveNode(state, id)
		end
		self:Prune(state)
	end

	-- Phase 2: connect must-haves
	self.phase = "connect"
	local ok, err = self:ConnectMustHaves(state)
	if not ok then
		return { error = err }
	end

	-- Phase 3: trim
	self.phase = "trim"
	ok, err = self:Trim(state, self.points)
	if not ok then
		return { error = err }
	end
	self:Track(state)

	-- Phase 4: fill
	self.phase = "fill"
	-- From scratch, a quick fill that measures only the targets that matter comes first, so there is a full tree early;
	-- the thorough fill that follows (see below) finds the tree the search then works on
	local quick = self.fromScratch
	local seedState = quick and self:CopyState(state)
	self.fillNormalReach = quick and FILL_NORMAL_REACH or nil
	self:Fill(state, self.points)
	self.fillNormalReach = nil
	self:Track(state)
	-- Phase 4b: repair a tree that misses its requirements
	local repaired
	if self:Evaluate(state).deficit > 0 then
		self.phase = "repair"
		state = self:Repair(state, self.points)
		self:Track(state)
		repaired = self.best
	end
	local fillTime = self:Now() - startTime
	local fillEvals = self.evalCount

	-- Phase 4c: the thorough fill, from the same start. It is kept to a share of the time limit, and only started when the
	-- quick one was cheap enough for it to fit; one that doesn't finish is dropped, and the best tree so far carries on.
	local thoroughTime, thoroughEvals
	if quick and not self.stopped and fillTime * 2 <= self.timeLimit * THOROUGH_FILL_SHARE then
		self.phase = "fill"
		self.deadline = startTime + self.timeLimit * THOROUGH_FILL_SHARE
		-- Starts from scratch: what the quick fill measured on a tiny tree undervalues what it skipped, and a node whose
		-- measurement came out low is never looked at again
		self.estimates = nil
		self.signatureRates, self.masteryRanks = nil, nil
		local trial = self:Fill(seedState, self.points)
		if self:Evaluate(trial).deficit > 0 and not self.stopped then
			self.phase = "repair"
			trial = self:Repair(trial, self.points)
		end
		self:Track(trial)
		if self.timeUp and not (self.shouldStop and self.shouldStop()) and self.evalCount < self.maxEvals then
			-- Only the share ran out, not the run
			self.stopped, self.timeUp = false, false
		end
		self.deadline = nil
		thoroughTime, thoroughEvals = self:Now() - startTime, self.evalCount
		repaired = nil
	end
	local afterFill = self:CopyState(quick and self.best or state)
	self.startState = afterFill

	-- The time limit covers the whole run; the fill always completes so there is a full tree to return
	local deadline = startTime + self.timeLimit
	self.deadline = deadline

	-- Phase 5 until time runs out. Model rounds come first and keep going while they find better trees or have
	-- measurements left to make. When one fails with nothing left to measure, or MODEL_MAX_FAILURES fail in a row, the
	-- model rests until the best tree changes and the real search takes over: Refine the best tree until its first
	-- improving move (then the model gets its turn on the new tree) or a local optimum, and from a local optimum, far
	-- swaps / perturbation followed by Refine (iterated local search).
	local modelRounds, modelFailures, modelRestingAt = 0, 0, nil
	local localOptima = 0
	local refinedKey
	local bestLosses, bestLossesFor = { }, nil
	local perturbedFor, perturbFailures
	local modelTime = 0
	local modelShare = self.modelShare or MODEL_SHARE
	-- From scratch, the fill's tree is refined before the model measures anything: the model's measuring is a large
	-- fixed cost that only pays off on a tree that is already decent
	local scratchRefined = not self.fromScratch
	while not self.stopped and self:Now() < deadline do
		local bestKey = stateKey(self.best)
		if self.best.eval.deficit > 0 and repaired ~= self.best then
			-- A best tree that misses its requirements gets one repair before anything else is tried on it
			self.phase = "repair"
			local trial = self:Repair(self:CopyState(self.best), self.points)
			repaired = self.best
			self:Track(trial)
			state = trial
		elseif not scratchRefined then
			scratchRefined = true
			self.phase = "refine"
			local refineDeadline = m_min(deadline, self:Now() + self.timeLimit * 0.25)
			local trial = self:Refine(self:CopyState(self.best), self.points, refineDeadline)
			self:Track(trial)
			state = trial
		elseif self.useModel and modelRestingAt ~= bestKey and modelTime <= modelShare * (self:Now() - startTime) then
			modelRounds = modelRounds + 1
			local roundStart = self:Now()
			local found, pending = self:ModelRound(deadline)
			modelTime = modelTime + self:Now() - roundStart
			if found then
				modelFailures = 0
			else
				modelFailures = modelFailures + 1
				if not pending or modelFailures >= MODEL_MAX_FAILURES then
					modelRestingAt = stateKey(self.best)
					modelFailures = 0
				end
			end
		elseif refinedKey ~= bestKey then
			self.phase = "refine"
			local trial = self:Refine(self:CopyState(self.best), self.points, deadline, true)
			self:Track(trial)
			state = trial
			if stateKey(self.best) == bestKey and not self.stopped and self:Now() < deadline then
				-- Nothing improved the best tree: it is a local optimum
				refinedKey = bestKey
				localOptima = localOptima + 1
			end
		else
			-- Work through far targets on the best tree first (each call tries ones not yet tried on it), then fall back
			-- to random perturbation once they run out
			if bestLossesFor ~= self.best then
				bestLosses, bestLossesFor = { }, self.best
			end
			self.phase = "far swap"
			local trial = self:FarSwap(self.best, self.points, self:BranchLossFunction(bestLosses))
			if not trial and not self.stopped then
				self.phase = "perturb"
				-- Kicks that keep failing to lead anywhere better grow, so a deep local optimum is eventually left
				if perturbedFor ~= self.best then
					perturbedFor, perturbFailures = self.best, 0
				end
				perturbFailures = perturbFailures + 1
				if perturbFailures % 3 == 0 then
					trial = self:PerturbRegion(self.best, self.points, 2 + m_floor(perturbFailures / 3) % 4)
				end
				trial = trial or self:Perturb(self.best, self.points, m_min(2 + m_floor(perturbFailures / 4), 8))
			end
			if trial and not self.stopped then
				self.phase = "refine"
				trial = self:Refine(trial, self.points, deadline)
				self:Track(trial)
				state = trial
				localOptima = localOptima + 1
			end
		end
	end

	local best = self.best or state
	return {
		localOptima = localOptima,
		modelRounds = modelRounds,
		initial = initial,
		initialEval = initialEval,
		afterFill = afterFill,
		afterFillEval = self:Evaluate(afterFill),
		best = best,
		bestEval = self:Evaluate(best),
		evals = self.evalCount,
		cacheHits = self.cacheHits,
		fillEvals = fillEvals,
		fillTime = fillTime,
		thoroughEvals = thoroughEvals,
		thoroughTime = thoroughTime,
		time = self:Now() - startTime,
		stopped = self.stopped and not self.timeUp,
	}
end

-- Lists nodes added and removed relative to the current spec
function TreeOptimizer:Diff(state)
	local added, removed = { }, { }
	for id, node in pairs(state.nodes) do
		local specNode = self.spec.allocNodes[id]
		if not specNode or (state.mastery[id] and state.mastery[id] ~= self.spec.masterySelections[id]) then
			t_insert(added, { id = id, node = self.spec.nodes[id], effectId = state.mastery[id] })
		end
	end
	for id, node in pairs(self.spec.allocNodes) do
		if not node.ascendancyName and node.type ~= "ClassStart" and node.type ~= "AscendClassStart" then
			if not state.nodes[id] or (state.mastery[id] and state.mastery[id] ~= self.spec.masterySelections[id]) then
				t_insert(removed, { id = id, node = node, effectId = self.spec.masterySelections[id] })
			end
		end
	end
	return added, removed
end

-- Undo state (as PassiveSpec:CreateUndoState) describing `state`, keeping the spec's ascendancy nodes
function TreeOptimizer:ToSpecState(state)
	local specState = self.spec:CreateUndoState()
	local hashList = { }
	for _, id in ipairs(specState.hashList) do
		local node = self.spec.nodes[id]
		if node and (node.ascendancyName or node.type == "ClassStart" or node.type == "AscendClassStart") then
			t_insert(hashList, id)
		end
	end
	for id in pairs(state.nodes) do
		t_insert(hashList, id)
	end
	specState.hashList = hashList
	local masteryEffects = { }
	for id, effectId in pairs(state.mastery) do
		masteryEffects[id] = effectId
	end
	specState.masteryEffects = masteryEffects
	return specState
end

-- Adds `state` to the build as a new tree and makes it active
function TreeOptimizer:ApplyToNewSpec(state, title)
	local oldSpec = self.spec
	local newSpec = new("PassiveSpec"):PassiveSpec(self.build, oldSpec.treeVersion)
	newSpec.title = title or ((oldSpec.title or "Default") .. " (optimized)")
	newSpec.jewels = copyTable(oldSpec.jewels)
	newSpec.optimizerMarks = copyTable(oldSpec.optimizerMarks or { })
	newSpec.optimizerRequirements = copyTable(oldSpec.optimizerRequirements or { })
	newSpec:RestoreUndoState(self:ToSpecState(state))
	newSpec:BuildClusterJewelGraphs()
	t_insert(self.build.treeTab.specList, newSpec)
	self.build.treeTab:SetActiveSpec(#self.build.treeTab.specList)
	return newSpec
end

return TreeOptimizer
