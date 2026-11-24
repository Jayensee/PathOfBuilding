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
--   5. Refine: remove a branch, refill without it, keep the result if it scored better. When no branch helps, swap a
--      travel node for a neighbour (Reroute), or take the path to a notable/keystone 5-12 points away and pay for it
--      with the cheapest branches anywhere in the tree (FarSwap).
--   6. Until time runs out, alternating:
--      - Model rounds: measure every node near the best tree once, search a cheap additive model of the score with
--        the moves above, and check its best trees with full calculations (see BuildModel).
--      - Far swaps on the best tree, then random perturbation, each followed by Refine.
--
local m_max = math.max
local m_min = math.min
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

-- Log-score cost of each missing attribute point. 10 missing points cost ~18% of the stat.
local ATTRIBUTE_PENALTY = 0.02
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
	self.onProgress = options.onProgress
	self.shouldStop = options.shouldStop
	self.useFullDPS = true
	self.useModel = options.useModel ~= false
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
	local omni = output.ReqOmni and output.ReqOmni > 0
	for _, attr in ipairs(omni and { "Omni" } or { "Str", "Dex", "Int" }) do
		local req = self.requirements[attr] or output["Req" .. attr] or 0
		local have = output[attr] or 0
		attributes[attr] = { req = req, have = have }
		if have < req then
			deficit = deficit + (req - have)
		end
	end
	local violations = { }
	if output.LifeUnreserved and output.LifeUnreserved < 1 and not output.ChaosInoculation then
		deficit = deficit + (1 - output.LifeUnreserved) * 0.5
		t_insert(violations, "unreserved life below 1")
	end
	local minMana = self.requirements.ManaUnreserved or 0
	if output.ManaUnreserved and output.ManaUnreserved < minMana then
		deficit = deficit + (minMana - output.ManaUnreserved) * 0.5
		t_insert(violations, minMana > 0 and ("unreserved mana below " .. minMana) or "unreserved mana negative")
	end
	for _, skillCost in ipairs(self.skillCosts) do
		local shortfall = self:SkillCostShortfall(skillCost, output)
		if shortfall > 0 then
			deficit = deficit + COST_WARNING_DEFICIT + shortfall * 0.5
			t_insert(violations, "can't pay " .. skillCost.pool:gsub("Unreserved", "") .. " cost of " .. skillCost.name)
		end
	end
	if self.sustainCost then
		for _, shortfall in ipairs(TreeOptimizer.SustainShortfalls(output)) do
			deficit = deficit + (1 - shortfall.recovery / shortfall.cost) * 100 * SUSTAIN_DEFICIT_PER_PERCENT
			t_insert(violations, string.format("main skill costs %.1f %s/s, recovering %.1f/s", shortfall.cost, shortfall.label, shortfall.recovery))
		end
	end
	return deficit, attributes, violations
end

-- Everything ConstraintDeficit reads, so the node value model can predict constraints from summed node effects
local CONSTRAINT_FIELDS = { "Str", "Dex", "Int", "Omni", "ReqStr", "ReqDex", "ReqInt", "ReqOmni", "Life", "LifeUnreserved",
	"LifeUnreservedPercent", "Mana", "ManaUnreserved", "ManaUnreservedPercent", "EnergyShield", "Rage",
	"ManaPerSecondCost", "ManaPercentPerSecondCost", "ESPerSecondCost", "ESPercentPerSecondCost", "LifePerSecondCost",
	"LifePercentPerSecondCost", "NetManaRegen", "ManaRegenRecovery", "ManaLeechGainRate", "NetEnergyShieldRegen",
	"EnergyShieldRegenRecovery", "EnergyShieldLeechGainRate", "EnergyShieldRecharge", "NetLifeRegen", "LifeRegenRecovery",
	"LifeLeechGainRate" }
local CONSTRAINT_FLAGS = { "ChaosInoculation", "EnergyShieldProtectsMana" }

function TreeOptimizer:ScoreFromOutput(output)
	local value = data.powerStatList.GetFromOutput(output, self.powerStat)
	local deficit, attributes, violations = self:ConstraintDeficit(output)
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
	end
	return {
		value = value,
		deficit = deficit,
		attributes = attributes,
		violations = violations,
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
		self.onProgress({ phase = self.phase, evals = self.evalCount, best = self.best and self.best.eval, elapsed = now() - (self.startTime or now()), timeLimit = self.timeLimit })
	end
	if (self.shouldStop and self.shouldStop()) or self.evalCount >= self.maxEvals then
		self.stopped = true
	end
	return result
end

-- Feasible trees always beat infeasible ones; otherwise compare the penalised score
function TreeOptimizer:IsBetter(a, b)
	if (a.deficit == 0) ~= (b.deficit == 0) then
		return a.deficit == 0
	end
	return a.score > b.score + EPSILON
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

-- Returns a list of { ids = {...}, size = n } for every node whose branch can be removed
function TreeOptimizer:RemovableBranches(state, maxSize)
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
-- Furthest away a mastery is considered. Ranking a mastery's effects takes a calculation per effect, and the path to a
-- mastery runs through a notable that is a candidate of its own; once that is taken, the mastery is next to the tree.
local FILL_MASTERY_REACH = 1

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
	-- How far paths may reach. Widened when nothing within reach helps (e.g. the class start is surrounded by
	-- nodes that don't affect the stat) so points aren't left unspent. Narrowed again after every accepted path:
	-- a wide frontier brings in many never-evaluated nodes after each step, which costs far more than re-widening.
	local reach = self.maxPathLength
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
			if node.type ~= "Mastery" or d <= masteryReach then
				local est = estimates[id]
				t_insert(candidates, { id = id, node = node, dist = d, priority = est and (est.gain / d), signature = self:NodeSignature(node), fresh = est and est.key == baseKey })
			end
		end
		local accepted
		while candidates[1] and not self.stopped do
			local best, bestPriority
			for _, cand in ipairs(candidates) do
				local priority = cand.priority or (useSignatures and signatureRates[cand.signature]) or math.huge
				if not best or priority > bestPriority then
					best, bestPriority = cand, priority
				end
			end
			if best.fresh then
				accepted = best
				break
			end
			-- Re-evaluate the most promising candidate against the current tree
			local path = buildPath(best.node, dist, parent)
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

-- Phase 5: refine by removing a branch and refilling without it

function TreeOptimizer:Refine(state, budget, deadline)
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
	while improved and not self.stopped and now() < deadline do
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
			if self.stopped or now() >= deadline then
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
		if not improved and not self.stopped and now() < deadline then
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

-- Node value model
--
-- A full calculation takes 10-25 ms, which limits the search to tens of thousands of trees. The model measures every
-- node near a base tree once (one calculation each: the calculator doesn't need nodes to be connected) and predicts any
-- tree as the base plus the sum of its changed nodes' effects, on the log of the stat and on every value the
-- constraints read. Predictions take microseconds, so the usual moves can search the model exhaustively; the trees it
-- picks are then checked with a full calculation. Measured on real builds, predictions stay within ~1% for trees up to
-- ~20 changed nodes and degrade gradually further out, mostly where stats interact (crit, conversion, thresholds).

-- How far from the base tree unallocated nodes are measured, and unallocated masteries (every effect costs a
-- calculation, and a mastery further out rarely makes the difference)
local MODEL_RADIUS = 6
local MODEL_MASTERY_RADIUS = 2
-- Distinct trees found on the model that are checked with full calculations per round
local MODEL_CHECKS = 6
-- Seconds the search on the model may take per round
local MODEL_SEARCH_TIME = 5
-- Model rounds pause after this many fail in a row, until the best tree has moved on (see Run)
local MODEL_MAX_FAILURES = 2

-- Effect of moving from `base` to `eval` on the modelled values
local function fieldDelta(base, fields)
	local delta = { }
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
	return delta
end

-- Measures the model around `base`: removing each allocated node, switching each allocated mastery to each other
-- effect, and adding each eligible node (and mastery effect) within MODEL_RADIUS. Returns nil if stopped part way.
function TreeOptimizer:BuildModel(base)
	self.captureFields = true
	local function measure(state)
		-- Cached results from before capturing started have no fields; recalculate those
		local key = stateKey(state)
		if self.scoreCache[key] and not self.scoreCache[key].fields then
			self.scoreCache[key] = nil
		end
		state.eval = nil
		return self:Evaluate(state).fields
	end
	-- add/switch are keyed by node id, or "id:effect" for masteries; measured[id] marks unallocated nodes with a value
	local model = { base = base, add = { }, remove = { }, switch = { }, measured = { } }
	model.baseFields = measure(self:CopyState(base))
	local function record(list, key, state)
		if not self.stopped then
			list[key] = fieldDelta(model.baseFields, measure(state))
		end
	end
	local ids = { }
	for id in pairs(base.nodes) do
		t_insert(ids, id)
	end
	t_sort(ids)
	for _, id in ipairs(ids) do
		if not self.mustHave[id] then
			local trial = self:CopyState(base)
			self:RemoveNode(trial, id)
			record(model.remove, id, trial)
		end
		local node = self.spec.nodes[id]
		if node.type == "Mastery" then
			for _, effectId in ipairs(self:AvailableEffects(base, node)) do
				if effectId ~= base.mastery[id] then
					local trial = self:CopyState(base)
					self:RemoveNode(trial, id)
					self:AddNode(trial, node, effectId)
					record(model.switch, id .. ":" .. effectId, trial)
				end
			end
		end
	end
	local dist = self:Frontier(base, MODEL_RADIUS)
	ids = { }
	for id in pairs(dist) do
		t_insert(ids, id)
	end
	t_sort(ids)
	for _, id in ipairs(ids) do
		local node = self.spec.nodes[id]
		if node.type == "Mastery" then
			if dist[id] <= MODEL_MASTERY_RADIUS then
				for _, effectId in ipairs(self:AvailableEffects(base, node)) do
					local trial = self:CopyState(base)
					self:AddNode(trial, node, effectId)
					record(model.add, id .. ":" .. effectId, trial)
					model.measured[id] = true
				end
			end
		else
			local trial = self:CopyState(base)
			self:AddNode(trial, node)
			record(model.add, id, trial)
			model.measured[id] = true
		end
	end
	self.captureFields = false
	if self.stopped then
		return nil
	end
	return model
end

-- Predicted evaluation of `state`, or nil if it uses a node the model didn't measure
function TreeOptimizer:ModelEvaluate(model, state)
	local fields = self:ModelFields(model, state)
	if not fields then
		return nil
	end
	local deficit, attributes, violations = self:ConstraintDeficit(fields)
	return {
		value = math.exp(fields.logValue),
		deficit = deficit,
		attributes = attributes,
		violations = violations,
		score = fields.logValue - deficit * ATTRIBUTE_PENALTY,
	}
end

-- Predicted values (log of the stat and everything the constraints read) of `state`, relative to the measured base
function TreeOptimizer:ModelFields(model, state)
	local base = model.base
	local fields = copyTable(model.baseFields, true)
	local function apply(delta)
		for key, value in pairs(delta) do
			fields[key] = (fields[key] or 0) + value
		end
	end
	for id in pairs(state.nodes) do
		local key = state.mastery[id] and (id .. ":" .. state.mastery[id]) or id
		local delta
		if not base.nodes[id] then
			delta = model.add[key]
		elseif state.mastery[id] and state.mastery[id] ~= base.mastery[id] then
			delta = model.switch[key]
		end
		if delta then
			apply(delta)
		elseif not base.nodes[id] or (state.mastery[id] and state.mastery[id] ~= base.mastery[id]) then
			return nil
		end
	end
	for id in pairs(base.nodes) do
		if not state.nodes[id] then
			if not model.remove[id] then
				return nil
			end
			apply(model.remove[id])
		end
	end
	return fields
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

-- One model round around the best tree: measure the model, search it with the usual moves, then check the most
-- promising trees it found with full calculations. Returns the best of those if it really beats the base, else nil.
function TreeOptimizer:ModelRound(deadline)
	local base = self:CopyState(self.best)
	local baseEval = self:Evaluate(base)
	self.phase = "node values"
	local model = self:BuildModel(base)
	if not model or now() >= deadline then
		return nil
	end
	-- A copy of the optimizer that scores trees with the model. It shares the graph and settings, but has its own
	-- caches and bookkeeping so nothing predicted leaks into the real search.
	self.phase = "model search"
	local unknown = { score = -math.huge, deficit = math.huge, value = 0, attributes = { }, violations = { } }
	local proxy = setmetatable({
		scoreCache = { }, estimates = { }, farEstimates = { }, farTried = { }, farFailures = { }, stopped = false,
	}, { __index = self })
	local predictions = 0
	proxy.Evaluate = function(p, state)
		if not state.eval then
			state.eval = self:ModelEvaluate(model, state) or unknown
			predictions = predictions + 1
			-- Predictions don't go through the real Evaluate, so report progress (letting the UI run) and honour Stop here
			if predictions % 50 == 0 then
				if self.onProgress then
					self.onProgress({ phase = self.phase, evals = self.evalCount, best = self.best and self.best.eval, elapsed = now() - self.startTime, timeLimit = self.timeLimit })
				end
				if self.shouldStop and self.shouldStop() then
					self.stopped = true
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
		return base.nodes[node.id] ~= nil or model.measured[node.id] == true
	end
	-- Candidates for checking: the best predicted tree within each distance of the base (predictions are most
	-- reliable near the base), and the best distinct trees overall
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
		local changed = changedNodes(state, base)
		local entry
		for i, limit in ipairs(limits) do
			if changed <= limit and (not bestWithin[i] or self:IsBetter(eval, bestWithin[i].eval)) then
				entry = entry or { state = p:CopyState(state), eval = eval, key = stateKey(state) }
				bestWithin[i] = entry
			end
		end
		if #top < MODEL_CHECKS or self:IsBetter(eval, top[#top].eval) then
			local key = stateKey(state)
			for _, other in ipairs(top) do
				if other.key == key then
					return
				end
			end
			t_insert(top, entry or { state = p:CopyState(state), eval = eval, key = key })
			t_sort(top, better)
			top[MODEL_CHECKS + 1] = nil
		end
	end

	-- Search the model: refine from the base, then perturb and refine (predictions are cheap, so this explores far
	-- more than the real search can)
	local searchDeadline = m_min(deadline, now() + MODEL_SEARCH_TIME)
	local trial = proxy:CopyState(base)
	trial.eval = nil
	proxy:Fill(trial, self.points)
	local current = proxy:Refine(trial, self.points, searchDeadline)
	while not proxy.stopped and now() < searchDeadline do
		local perturbed = proxy:Perturb(current, self.points)
		perturbed = proxy:Refine(perturbed, self.points, searchDeadline)
		if self:IsBetter(proxy:Evaluate(perturbed), proxy:Evaluate(current)) then
			current = perturbed
		end
	end

	-- Check candidates with full calculations, nearest to the base first
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
	local checked, checks = { }, 0
	for _, candidate in ipairs(candidates) do
		if self.stopped or checks >= MODEL_CHECKS then
			break
		end
		if not checked[candidate.key] then
			checked[candidate.key] = true
			checks = checks + 1
			local state = self:CopyState(candidate.state)
			state.eval = nil
			local eval = self:Evaluate(state)
			self:Track(state)
			if self:IsBetter(eval, bestEval or baseEval) then
				bestState, bestEval = state, eval
			end
		end
	end
	return bestState
end

-- Removes a few random branches from `state` and refills without them, to escape a local optimum
function TreeOptimizer:Perturb(state, budget)
	local trial = self:CopyState(state)
	local banned = { }
	local count = self.random(2, 4)
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

-- Main entry point. Returns a result table; call :ApplyToNewSpec(result) to add it to the build.
function TreeOptimizer:Run()
	local startTime = now()
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
	self:Fill(state, self.points)
	self:Track(state)
	local afterFill = self:CopyState(state)
	local fillTime = now() - startTime
	local fillEvals = self.evalCount

	-- The time limit covers the whole run; the fill always completes so there is a full tree to return
	local deadline = startTime + self.timeLimit

	-- Phase 5 until time runs out: model rounds, refine, then far swaps / perturbation and refine again (iterated local
	-- search). Model rounds and the real search find different improvements, so they alternate: a round runs first, again
	-- straight after every successful round (once its result is refined), and otherwise whenever the search has moved the
	-- best tree on by 1% since the last round. Rounds stop for the run after MODEL_MAX_FAILURES fail in a row.
	local modelRounds, modelFailures, modelScore, modelSucceeded = 0, 0, nil, false
	local localOptima = 0
	local bestLosses, bestLossesFor = { }, nil
	while not self.stopped and now() < deadline do
		local trial
		if self.useModel and modelFailures < MODEL_MAX_FAILURES and (modelSucceeded or not modelScore or self.best.eval.score > modelScore + 0.01) then
			modelRounds = modelRounds + 1
			modelScore = self.best.eval.score
			trial = self:ModelRound(deadline)
			modelSucceeded = trial ~= nil
			modelFailures = trial and 0 or (modelFailures + 1)
		end
		if not trial and localOptima == 0 then
			-- Refine the filled tree itself first
			trial = self:CopyState(self.best)
		end
		if not trial and not self.stopped then
			-- Work through far targets on the best tree first (each call tries ones not yet tried on it), then fall back
			-- to random perturbation once they run out
			if bestLossesFor ~= self.best then
				bestLosses, bestLossesFor = { }, self.best
			end
			self.phase = "far swap"
			trial = self:FarSwap(self.best, self.points, self:BranchLossFunction(bestLosses))
			if not trial and not self.stopped then
				self.phase = "perturb"
				trial = self:Perturb(self.best, self.points)
			end
		end
		if trial then
			self.phase = "refine"
			trial = self:Refine(trial, self.points, deadline)
			self:Track(trial)
			state = trial
			localOptima = localOptima + 1
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
		time = now() - startTime,
		stopped = self.stopped,
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
