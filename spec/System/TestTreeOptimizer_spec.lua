describe("TreeOptimizer", function()
	local TreeOptimizer
	local lifeStat = { stat = "Life", label = "Life" }

	before_each(function()
		newBuild()
		TreeOptimizer = LoadModule("Modules/TreeOptimizer")
	end)

	local function run(options)
		options.powerStat = options.powerStat or lifeStat
		options.timeLimit = options.timeLimit or 5
		options.maxEvals = options.maxEvals or 400
		local optimizer = TreeOptimizer.new(build, options)
		local result = optimizer:Run()
		assert.is_nil(result.error)
		return optimizer, result
	end

	local function assertConnected(optimizer, state)
		local reached = optimizer:Reachable(state.nodes)
		for id in pairs(state.nodes) do
			assert.is_true(reached[id] == true, "node " .. id .. " is not connected to the class start")
		end
	end

	-- Closest unallocated notable at least `minDist` points away
	local function findNotable(minDist)
		local best
		for _, node in pairs(build.spec.nodes) do
			if node.type == "Notable" and not node.ascendancyName and node.pathDist and node.pathDist >= minDist and node.pathDist < 1000 then
				if not best or node.pathDist < best.pathDist then
					best = node
				end
			end
		end
		return best
	end

	describe("evaluation clock", function()
		local function fingerprint(optimizer, result)
			local ids = { }
			for id in pairs(result.best.nodes) do
				table.insert(ids, id)
			end
			table.sort(ids)
			return table.concat(ids, ",") .. string.format("|%.17g|%d", result.bestEval.score, optimizer.evalCount)
		end

		it("gives identical results for the same seed, whatever the machine does", function()
			local options = { clock = "evals", evalTime = 0.015, timeLimit = 1.2, maxEvals = math.huge, points = 12, fromScratch = true, seed = 3, logPredictions = true }
			local first, firstResult = run(copyTable(options))
			-- Burn CPU in between: real time must not matter
			local spin = os.clock()
			while os.clock() - spin < 0.3 do end
			local second, secondResult = run(copyTable(options))
			assert.are.equal(fingerprint(first, firstResult), fingerprint(second, secondResult))
			assert.are.equal(#first.predictionLog, #second.predictionLog)
			-- The budget is counted in calculations (1.2s at 15ms is 80, shared with the model search)
			assert.is_true(first.evalCount > 10 and first.evalCount < 80 + 40)
		end)
	end)

	describe("distance penalty", function()
		it("learns the model's optimism per distance from its checks and subtracts it", function()
			local optimizer = run({ points = 12, fromScratch = true, timeLimit = 0.01 })
			assert.are.equal(0, optimizer:DistancePenalty(40))
			-- Predictions 0.01 too high within 5 nodes, 0.1 too high within 6-10, 0.4 beyond 20
			for _, case in ipairs({ { 3, 0.01 }, { 8, 0.1 }, { 40, 0.4 } }) do
				for _ = 1, 20 do
					optimizer:RecordCheck({ rawScore = 1 + case[2], deficit = 0, changed = case[1] }, { score = 1, deficit = 0 })
				end
			end
			local small, middle, large = optimizer:DistancePenalty(3), optimizer:DistancePenalty(8), optimizer:DistancePenalty(40)
			assert.is_true(small > 0 and small < middle and middle < large)
			assert.is_true(optimizer:DistancePenalty(60) > large)
			assert.are.near(0.01 * 20 / 23, small, 1e-9)
		end)
	end)

	describe("structured model", function()
		-- A score function the optimizer knows nothing about: it reads Life and FireResist, nothing else
		local function score(output)
			return output.Life * (100 + (output.FireResist or 0)) / 100
		end

		it("discovers the fields a custom score function reads, and predicts through it", function()
			local optimizer = run({ points = 12, fromScratch = true, scoreFunction = score, structuredModel = true, adaptiveSchedule = false, clock = "evals", timeLimit = 1.5, maxEvals = math.huge })
			assert.is_true(optimizer.structured)
			assert.are.same({ "FireResist", "Life" }, optimizer.discPaths)
			local model = optimizer.model
			assert.is_truthy(model)
			-- On the model's own base the predicted output is the base output, so the prediction is its real score
			local predicted = optimizer:PredictLogValue(model, model.baseFields, { }, { })
			assert.is_truthy(predicted)
			assert.are.near(optimizer:Evaluate(model.base).fields.logValue, predicted, 1e-9)
			-- Measured entries carry changes of exactly the discovered fields
			local sawLog = false
			for _, entry in pairs(model.with) do
				for path in pairs(entry.dl or { }) do
					assert.is_true(optimizer.discSet[path])
					sawLog = true
				end
			end
			assert.is_true(sawLog)
		end)

		it("learns per field from its checks how far to trust the summed change, and clamps to what was observed", function()
			local optimizer = run({ points = 12, fromScratch = true, scoreFunction = score, structuredModel = true, adaptiveSchedule = false, clock = "evals", timeLimit = 1.5, maxEvals = math.huge })
			assert.is_true(optimizer.fieldCalibration)
			local model = optimizer.model
			local path = "Life"
			local base = model.baseFields[path]
			assert.is_true(base > 0)
			assert.are.equal(1, optimizer:FieldFactor(path))
			-- A saturating field: the real log change is half of the predicted one, and never goes below 0.5 * base
			local function real(x)
				return math.max(0.5 * base, base * math.exp(x / 2))
			end
			local function error(x)
				local sum = x
				local predicted = base * math.exp(sum * optimizer:FieldFactor(path))
				predicted = optimizer:FieldClamp(path, predicted, base)
				return math.abs(math.log(predicted / real(x)))
			end
			local before = error(-3)
			local beforeUp = error(1.5)
			for _, x in ipairs({ -0.4, 0.3, -0.8, 0.6, -1.2, 1.0, -2, 0.2 }) do
				optimizer:RecordFieldCheck({ rawPred = { [path] = base * math.exp(x) }, basePaths = { [path] = base } }, { [path] = real(x) })
			end
			local factor = optimizer:FieldFactor(path)
			assert.is_true(factor > 0.4 and factor < 0.8)
			-- Far past what was seen the calibrated prediction is closer, in both directions
			assert.is_true(error(-3) < before)
			assert.is_true(error(1.5) < beforeUp)
			-- Without calibration nothing is learned
			local plain = run({ points = 12, fromScratch = true, scoreFunction = score, structuredModel = true, fieldCalibration = false, timeLimit = 0.01 })
			assert.is_false(plain.fieldCalibration)
		end)

		it("falls back to the stat's own log when the structured model is off", function()
			local optimizer = run({ points = 12, fromScratch = true, scoreFunction = score, structuredModel = false })
			assert.is_false(optimizer.structured)
			assert.are.same({ }, optimizer.discPaths)
		end)
	end)

	it("fills the budget with a connected tree that improves the stat", function()
		local optimizer, result = run({ points = 12, fromScratch = true })
		assert.is_true(result.best.count <= 12)
		assert.is_true(result.best.count > 0)
		assert.is_true(result.bestEval.value > result.initialEval.value)
		assertConnected(optimizer, result.best)
	end)

	it("keeps must-have nodes and avoids must-not nodes", function()
		local mustHave = assert(findNotable(5))
		local optimizer, result = run({ points = 12, fromScratch = true, mustHave = { [mustHave.id] = true } })
		assert.is_truthy(result.best.nodes[mustHave.id])
		assertConnected(optimizer, result.best)

		-- Ban everything the unconstrained tree took (except the must-have path) and check none of it comes back
		local mustNot = { }
		for id in pairs(result.best.nodes) do
			if id ~= mustHave.id and build.spec.nodes[id].type == "Notable" then
				mustNot[id] = true
			end
		end
		local optimizer2, result2 = run({ points = 12, fromScratch = true, mustHave = { [mustHave.id] = true }, mustNot = mustNot })
		assert.is_truthy(result2.best.nodes[mustHave.id])
		for id in pairs(mustNot) do
			assert.is_nil(result2.best.nodes[id])
		end
		assertConnected(optimizer2, result2.best)
	end)

	it("reports an error when a must-have is unreachable", function()
		local target = assert(findNotable(3))
		-- Wall off the target by banning every neighbour
		local mustNot = { }
		for _, other in ipairs(target.linked) do
			mustNot[other.id] = true
		end
		local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat, points = 20, fromScratch = true, mustHave = { [target.id] = true }, mustNot = mustNot })
		local result = optimizer:Run()
		assert.is_truthy(result.error)
	end)

	it("treats manual attribute requirements as constraints", function()
		local _, result = run({ points = 15, fromScratch = true, requirements = { Str = 60 } })
		local str = result.bestEval.attributes.Str
		assert.are.equal(60, str.req)
		assert.is_true(str.have >= 60)
		assert.are.equal(0, result.bestEval.deficit)
	end)

	it("repairs a tree that misses its requirements by swapping nodes within the budget", function()
		-- Fill for life only, then demand more Strength than that tree has
		local probe = TreeOptimizer.new(build, { powerStat = lifeStat, points = 15, fromScratch = true })
		local state = probe:NewState()
		probe:Fill(state, 15)
		local str = probe:Evaluate(state).attributes.Str.have
		local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat, points = 15, fromScratch = true, requirements = { Str = str + 25 } })
		state.eval = nil
		local before = optimizer:Evaluate(state)
		assert.is_true(before.deficit > 0)
		assert.is_truthy(before.needs.Str)
		local after = optimizer:Repair(state, 15)
		local eval = optimizer:Evaluate(after)
		assert.are.equal(0, eval.deficit)
		assert.is_true(eval.attributes.Str.have >= str + 25)
		assert.is_true(after.count <= 15)
		assertConnected(optimizer, after)
		-- A tree that meets its requirements is left alone
		assert.are.equal(after, optimizer:Repair(after, 15))
	end)

	it("prefers trees closer to meeting their requirements", function()
		local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat })
		local valid = { deficit = 0, score = 1 }
		local nearly = { deficit = 1, score = 2 }
		local far = { deficit = 30, score = 3 }
		assert.is_true(optimizer:IsBetter(valid, nearly))
		assert.is_true(optimizer:IsBetter(nearly, far))
		assert.is_false(optimizer:IsBetter(far, nearly))
	end)

	it("re-ranks a mastery's effects only after the tree has changed enough", function()
		local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat, fromScratch = true })
		local mastery
		for _, node in pairs(build.spec.nodes) do
			if node.type == "Mastery" and #optimizer:GetMasteryEffects(node) >= 4 and (not mastery or node.id < mastery.id) then
				mastery = node
			end
		end
		local state = TreeOptimizer.new(build, { powerStat = lifeStat, fromScratch = true }):NewState()
		TreeOptimizer.new(build, { powerStat = lifeStat, fromScratch = true }):Fill(state, 10)
		state.eval = nil
		local evals = optimizer.evalCount
		local first = optimizer:BestMasteryEffect(state, mastery)
		assert.are.equal(#optimizer:AvailableEffects(state, mastery), optimizer.evalCount - evals)
		-- One more node: only the best effects of the last ranking are evaluated again
		local grown = optimizer:CopyState(state)
		for id in pairs(optimizer:Frontier(state, 1)) do
			if build.spec.nodes[id].type ~= "Mastery" then
				optimizer:AddNode(grown, build.spec.nodes[id])
				break
			end
		end
		assert.are.equal(state.count + 1, grown.count)
		evals = optimizer.evalCount
		optimizer:BestMasteryEffect(grown, mastery)
		assert.is_true(optimizer.evalCount - evals <= 2)
		assert.is_truthy(first)
	end)

	it("keeps the minimum unreserved mana", function()
		local minMana = math.floor(build.calcsTab.mainOutput.ManaUnreserved) + 30
		local optimizer, result = run({ points = 20, fromScratch = true, requirements = { ManaUnreserved = minMana } })
		assert.are.equal(0, result.bestEval.deficit)
		optimizer:ApplyToNewSpec(result.best)
		build.buildFlag = true
		runCallback("OnFrame")
		assert.is_true(build.calcsTab.mainOutput.ManaUnreserved >= minMana)
	end)

	describe("multi-point moves", function()
		-- Greedy tree built from single node steps, so notables a few points away are never taken
		local function shortSightedTree(points, clock)
			local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat, points = points, fromScratch = true, maxPathLength = 1, clock = clock })
			local state = optimizer:NewState()
			optimizer:Fill(state, points)
			return optimizer, state
		end
		local function assertValidSwap(optimizer, before, after, points)
			assert.is_true(after.count <= points)
			assertConnected(optimizer, after)
			assert.is_true(optimizer:IsBetter(optimizer:Evaluate(after), optimizer:Evaluate(before)))
		end

		it("swaps cheap branches for the path to a notable further than a fill step reaches", function()
			local optimizer, state = shortSightedTree(15)
			local after = optimizer:FarSwap(state, 15, optimizer:BranchLossFunction({ }))
			assert.is_truthy(after)
			assertValidSwap(optimizer, state, after, 15)
			local gotFarNode = false
			for id, node in pairs(after.nodes) do
				if not state.nodes[id] and node.type ~= "Normal" and node.type ~= "Mastery" then
					gotFarNode = true
				end
			end
			assert.is_true(gotFarNode)
		end)

		it("kicks with far anchors: connected, within budget, must-haves restored, deterministic", function()
			local function kick()
				local optimizer, state = shortSightedTree(40, "evals")
				optimizer.timeLimit = 5
				optimizer.maxEvals = 800
				local mustBefore = copyTable(optimizer.mustHave, true)
				local after = optimizer:AnchorKick(state, 40, optimizer:BranchLossFunction({ }))
				assert.is_truthy(after)
				assert.is_true(after.count <= 40)
				assertConnected(optimizer, after)
				assert.are.same(mustBefore, optimizer.mustHave)
				assert.is_truthy(optimizer.anchorKicks[1])
				return optimizer:Evaluate(after).score
			end
			assert.are.equal(kick(), kick())
		end)

		it("kicks with a cap on its time spend less, with conditional anchors on or off, and stay valid and deterministic", function()
			local function kick(opts, capped)
				local optimizer, state = shortSightedTree(40, "evals")
				for k, v in pairs(opts) do optimizer[k] = v end
				optimizer.timeLimit = 5
				optimizer.maxEvals = 3000
				optimizer.startTime = optimizer:Now()
				local start = optimizer.evalCount
				local after = optimizer:AnchorKick(state, 40, optimizer:BranchLossFunction({ }), capped and (optimizer:Now() + 0.1) or nil)
				if after then
					assert.is_true(after.count <= 40)
					assertConnected(optimizer, after)
				end
				return optimizer.evalCount - start, after and optimizer:Evaluate(after).score
			end
			local freeEvals = kick({ anchorConditional = true }, false)
			local cappedEvals = kick({ anchorConditional = true }, true)
			assert.is_true(cappedEvals < freeEvals)
			local a1, a2 = kick({ anchorConditional = true, anchorPathValue = false }, false)
			assert.is_true(a1 > 0)
			local b1, b2 = kick({ anchorConditional = true, anchorPathValue = false }, false)
			assert.are.equal(a2, b2)
			kick({ anchorConditional = false, anchorPathValue = true }, false)
		end)

		it("stops a kick at once: no calculations past the stop, tree connected and within budget, must-haves restored", function()
			for _, limit in ipairs({ 17, 30, 45, 76 }) do
				local optimizer, state = shortSightedTree(40, "evals")
				optimizer.timeLimit = 5
				optimizer.startTime = optimizer:Now()
				local mustBefore = copyTable(optimizer.mustHave, true)
				local start = optimizer.evalCount
				optimizer.shouldStop = function() return optimizer.evalCount - start >= limit end
				local after = optimizer:AnchorKick(state, 40, optimizer:BranchLossFunction({ }))
				assert.is_true(optimizer.evalCount - start <= limit + 3, "calculations past the stop at " .. limit)
				if after then
					assert.is_true(after.count <= 40)
					assertConnected(optimizer, after)
				end
				assert.are.same(mustBefore, optimizer.mustHave)
			end
		end)

		it("runs with the anchor kicks switched off or at every plateau", function()
			local _, off = run({ anchorShare = 0, anchorPlateau = 0, clock = "evals", maxEvals = 300 })
			local _, on = run({ anchorShare = 1, anchorPlateau = 1, anchorSlice = 0.05, clock = "evals", maxEvals = 300 })
			assert.is_truthy(off.best)
			assert.is_truthy(on.best)
		end)

		it("runs with the adaptive schedule: records every method's turns, deterministic, within budget", function()
			local function adaptive()
				local optimizer, result = run({ adaptiveSchedule = true, adaptiveStart = 0.1, adaptiveSlice = 0.1, clock = "evals", timeLimit = 6, maxEvals = 400, seed = 5 })
				assert.is_true(result.evals <= 400)
				assert.is_truthy(result.schedule.refine)
				return optimizer:Evaluate(result.best).score, result.schedule
			end
			local scoreA, scheduleA = adaptive()
			local scoreB, scheduleB = adaptive()
			assert.are.equal(scoreA, scoreB)
			for name, arm in pairs(scheduleA) do
				assert.are.equal(arm.pulls, scheduleB[name].pulls)
			end
			-- The method with too little of the run's time is picked first, whatever the rates
			local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat, adaptiveMinShare = 0.05 })
			local arms = { refine = { time = 10, gain = 5, recentTime = 10, recentGain = 5 } }
			assert.are.equal("far", optimizer:PickArm(arms, { refine = true, far = true }, 1))
			assert.are.equal("refine", optimizer:PickArm(arms, { refine = true }, 1))
		end)

		it("predicts single node changes exactly with the node value model", function()
			local optimizer, state = shortSightedTree(10)
			optimizer.points = 10
			local model = optimizer:BuildModel(state)
			assert.is_truthy(model)
			local function close(a, b)
				return math.abs(a - b) < 1e-6 * math.max(1, math.abs(b))
			end
			assert.is_true(close(optimizer:ModelEvaluate(model, state).score, optimizer:Evaluate(state).score))
			-- Removing a leaf and adding a frontier node, each on their own, match a full calculation
			local branch = optimizer:RemovableBranches(state, 1)[1]
			local removed = optimizer:CopyState(state)
			optimizer:RemoveNode(removed, branch.ids[1])
			local addId
			for id in pairs(optimizer:Frontier(state, 1)) do
				if build.spec.nodes[id].type ~= "Mastery" and (not addId or id < addId) then
					addId = id
				end
			end
			local added = optimizer:CopyState(state)
			optimizer:AddNode(added, build.spec.nodes[addId])
			for _, trial in ipairs({ removed, added }) do
				local predicted = optimizer:ModelEvaluate(model, trial)
				trial.eval = nil
				assert.is_true(close(predicted.score, optimizer:Evaluate(trial).score))
			end
		end)

		it("finds the same removable branches in one pass as with a search per node", function()
			local optimizer, state = shortSightedTree(25)
			local _, filled = run({ points = 30, fromScratch = true })
			for _, tree in ipairs({ state, filled.best }) do
				for _, maxSize in ipairs({ 1, 3, 6, math.huge }) do
					local function describe(branches)
						local list = { }
						for _, branch in ipairs(branches) do
							local ids = copyTable(branch.ids)
							table.sort(ids)
							table.insert(list, branch.key .. "=" .. table.concat(ids, ","))
						end
						return table.concat(list, ";")
					end
					assert.is_truthy(optimizer:BranchCuts(tree))
					local fast = describe(optimizer:RemovableBranches(tree, maxSize))
					-- An Intuitive Leap-like makes RemovableBranches fall back to one search per node
					local leapNodes = optimizer.leapNodes
					optimizer.leapNodes = { { id = -1, intuitiveLeapLikesAffecting = { } } }
					local slow = describe(optimizer:RemovableBranches(tree, maxSize))
					optimizer.leapNodes = leapNodes
					assert.are.equal(slow, fast)
				end
			end
		end)

		it("keeps nodes next to the class start connected when a loop closes through it", function()
			local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat, points = 30, fromScratch = true })
			local root = optimizer.root
			-- Shortest path between two nodes next to the class start that avoids the class start
			local state, found = optimizer:NewState(), false
			for _, first in ipairs(root.linked) do
				for _, last in ipairs(root.linked) do
					if not found and first.id < last.id and optimizer:IsEligible(first) and optimizer:IsEligible(last) then
						local parent, queue, head = { [first.id] = false }, { first }, 1
						while queue[head] and not parent[last.id] do
							local node = queue[head]
							head = head + 1
							for _, other in ipairs(node.linked) do
								if parent[other.id] == nil and other ~= root and other.type ~= "Mastery" and optimizer:IsEligible(other) then
									parent[other.id] = node
									table.insert(queue, other)
								end
							end
						end
						if parent[last.id] and optimizer:NewState() then
							local node = last
							while node do
								optimizer:AddNode(state, node)
								node = parent[node.id]
							end
							found = true
						end
					end
				end
			end
			assert.is_true(found)
			assert.is_truthy(optimizer:BranchCuts(state))
			local function describe(branches)
				local list = { }
				for _, branch in ipairs(branches) do
					local ids = copyTable(branch.ids)
					table.sort(ids)
					table.insert(list, branch.key .. "=" .. table.concat(ids, ","))
				end
				return table.concat(list, ";")
			end
			local fast = describe(optimizer:RemovableBranches(state, math.huge))
			local leapNodes = optimizer.leapNodes
			optimizer.leapNodes = { { id = -1, intuitiveLeapLikesAffecting = { } } }
			local slow = describe(optimizer:RemovableBranches(state, math.huge))
			optimizer.leapNodes = leapNodes
			assert.are.equal(slow, fast)
		end)

		it("measures nodes with the same stats once, with exact predictions for each", function()
			local optimizer, state = shortSightedTree(10)
			optimizer.points = 10
			local evalsBefore = optimizer.evalCount
			local model = optimizer:BuildModel(state)
			-- Two unallocated nodes with the same stats: one was measured, the other reuses its measurement
			local bySignature, pair = { }, nil
			for id in pairs(optimizer:Frontier(state, 6)) do
				local node = build.spec.nodes[id]
				if node.type ~= "Mastery" then
					local signature = optimizer:ModelSignature(model, { id = id })
					if signature and bySignature[signature] then
						pair = { bySignature[signature], id }
						break
					elseif signature then
						bySignature[signature] = id
					end
				end
			end
			assert.is_truthy(pair)
			assert.are.equal(model.with[pair[1]].delta, model.with[pair[2]].delta)
			local tasks = 0
			for _ in pairs(model.with) do
				tasks = tasks + 1
			end
			assert.is_true(optimizer.evalCount - evalsBefore < tasks)
			for _, id in ipairs(pair) do
				local trial = optimizer:CopyState(state)
				optimizer:AddNode(trial, build.spec.nodes[id])
				local predicted = optimizer:ModelEvaluate(model, trial)
				trial.eval = nil
				assert.is_true(math.abs(predicted.score - optimizer:Evaluate(trial).score) < 1e-6)
			end
		end)

		it("doesn't measure the model past the deadline", function()
			local optimizer, state = shortSightedTree(10)
			optimizer.points = 10
			optimizer.startTime = os.clock()
			optimizer:Track(state)
			local evalsBefore = optimizer.evalCount
			assert.is_nil(optimizer:ModelRound(os.clock() - 1))
			assert.are.equal(evalsBefore, optimizer.evalCount)
		end)

		it("only returns model round results that a full calculation confirms", function()
			local optimizer, state = shortSightedTree(15)
			optimizer.points = 15
			optimizer.startTime = os.clock()
			optimizer:Track(state)
			local before = optimizer:Evaluate(state)
			local after = optimizer:ModelRound(math.huge)
			if after then
				assert.is_true(after.count <= 15)
				assertConnected(optimizer, after)
				after.eval = nil
				assert.is_true(optimizer:IsBetter(optimizer:Evaluate(after), before))
			end
		end)

		it("only reroutes to trees that stay connected and score better", function()
			local optimizer, state = shortSightedTree(15)
			local after = optimizer:Reroute(state)
			if after then
				assert.are.equal(state.count, after.count)
				assertValidSwap(optimizer, state, after, 15)
			end
		end)
	end)

	describe("main skill cost sustain", function()
		-- A level 1 Witch casting Fireball: costs ~7 mana/s against 1 mana/s of regeneration, and nothing near the
		-- Witch start gives life, so maximising life alone neglects mana recovery
		local function fireballWitch()
			build.spec:SelectClass(3)
			build.skillsTab:PasteSocketGroup("Fireball 1/0  1\n")
			build.buildFlag = true
			runCallback("OnFrame")
			runCallback("OnFrame")
		end
		local function appliedSustain(optimizer, result)
			optimizer:ApplyToNewSpec(result.best)
			build.buildFlag = true
			runCallback("OnFrame")
			local out = build.calcsTab.mainOutput
			return out.ManaPerSecondCost, out.ManaRegenRecovery + (out.ManaLeechGainRate or 0)
		end

		it("requires recovery to keep up with the main skill's cost per second", function()
			fireballWitch()
			local probe = TreeOptimizer.new(build, { powerStat = lifeStat })
			local initial = probe:Evaluate(probe:NewState())
			assert.is_true(initial.deficit > 0)
			assert.is_truthy(initial.violations[1]:match("main skill costs"))

			local optimizer, result = run({ points = 30, maxEvals = 300 })
			assert.are.equal(0, result.bestEval.deficit)
			local cost, recovery = appliedSustain(optimizer, result)
			assert.is_true(cost > 0)
			assert.is_true(recovery >= cost)
		end)

		it("can be turned off", function()
			fireballWitch()
			local optimizer, result = run({ points = 30, maxEvals = 600, requirements = { SustainCost = 0 } })
			assert.is_false(optimizer.sustainCost)
			assert.are.equal(0, result.bestEval.deficit)
			-- Every point is spent even though nothing within the usual path length adds life
			assert.are.equal(30, result.best.count)
			local cost, recovery = appliedSustain(optimizer, result)
			assert.is_true(recovery < cost)
		end)

		it("pools mana and energy shield recovery under Eldritch Battery", function()
			local shortfalls = TreeOptimizer.SustainShortfalls({ ManaPerSecondCost = 0, ESPerSecondCost = 50, ManaRegenRecovery = 20, EnergyShieldRegenRecovery = 10, EnergyShieldRecharge = 0 })
			assert.are.equal(1, #shortfalls)
			assert.are.equal("ES", shortfalls[1].label)
			assert.are.equal(10, shortfalls[1].recovery)

			shortfalls = TreeOptimizer.SustainShortfalls({ ESPerSecondCost = 50, ManaRegenRecovery = 20, EnergyShieldRegenRecovery = 10, EnergyShieldLeechGainRate = 25, EnergyShieldProtectsMana = true })
			assert.are.equal(0, #shortfalls)

			-- Net regen (after degens) replaces gross regen, and percent costs scale with the pool
			shortfalls = TreeOptimizer.SustainShortfalls({ ManaPercentPerSecondCost = 2, Mana = 1000, ManaRegenRecovery = 30, NetManaRegen = 15, ManaLeechGainRate = 4 })
			assert.are.equal(1, #shortfalls)
			assert.are.equal(20, shortfalls[1].cost)
			assert.are.equal(19, shortfalls[1].recovery)
		end)
	end)

	describe("jewel sockets", function()
		-- Allocates the shortest path to the nearest jewel socket and puts `item` in it
		local function socketJewel(item)
			local spec = build.spec
			local start = spec.nodes[spec.curClass.startNodeId]
			local queue, parent, head = { start }, { [start.id] = false }, 1
			local socket
			while queue[head] do
				local node = queue[head]
				head = head + 1
				if node.isJewelSocket then
					socket = node
					break
				end
				for _, other in ipairs(node.linked) do
					if parent[other.id] == nil and other.type ~= "Mastery" and not other.ascendancyName then
						parent[other.id] = node
						table.insert(queue, other)
					end
				end
			end
			local node = socket
			while node do
				node.alloc = true
				spec.allocNodes[node.id] = node
				node = parent[node.id] or nil
			end
			build.itemsTab:AddItem(item, true)
			spec.jewels[socket.id] = item.id
			build.itemsTab.sockets[socket.id].selItemId = item.id
			spec:BuildAllDependsAndPaths()
			build.buildFlag = true
			-- The newly socketed jewel only shows up in the main output on the second pass
			runCallback("OnFrame")
			runCallback("OnFrame")
			return socket
		end

		local function currentState(optimizer)
			local state = optimizer:NewState()
			for _, node in pairs(build.spec.allocNodes) do
				if not node.ascendancyName and node.type ~= "ClassStart" and node.type ~= "AscendClassStart" then
					optimizer:AddNode(state, node)
				end
			end
			return state
		end

		local function lifeJewel()
			return new("Item"):Item("Rarity: RARE\nTest Spark\nCrimson Jewel\nImplicits: 0\n+200 to maximum Life\n")
		end

		it("treats sockets as regular nodes that carry their jewel", function()
			local socket = socketJewel(lifeJewel())
			local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat })
			assert.is_nil(next(optimizer.lockedSockets))
			assert.is_nil(optimizer.mustHave[socket.id])

			local state = currentState(optimizer)
			local withJewel = optimizer:Evaluate(state).value
			assert.are.near(build.calcsTab.mainOutput.Life, withJewel, 0.5)
			local without = optimizer:CopyState(state)
			optimizer:RemoveNode(without, socket.id)
			assert.is_true(withJewel - optimizer:Evaluate(without).value >= 200)

			-- Dropping the socket evaluates the same as a tree that really doesn't have it
			local predicted = optimizer:Evaluate(without).value
			optimizer:ApplyToNewSpec(without)
			build.buildFlag = true
			runCallback("OnFrame")
			assert.are.near(predicted, build.calcsTab.mainOutput.Life, 0.5)
		end)

		it("leaves out unallocated sockets whose jewel is socketed elsewhere", function()
			local socket = socketJewel(lifeJewel())
			local otherSocket
			for _, node in pairs(build.spec.nodes) do
				if node.isJewelSocket and not node.alloc and not node.expansionJewel then
					otherSocket = node
					break
				end
			end
			build.spec.jewels[otherSocket.id] = build.spec.jewels[socket.id]
			local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat })
			assert.is_true(optimizer.mustNot[otherSocket.id])
			assert.is_truthy(optimizer.lockedSockets[otherSocket.id])
			assert.is_nil(optimizer.lockedSockets[socket.id])
		end)

		it("reverts nodes conquered by a timeless jewel when its socket is dropped", function()
			local socket = socketJewel(new("Item"):Item("Rarity: UNIQUE\nLethal Pride\nTimeless Jewel\nRadius: Large\nImplicits: 0\nCommanded leadership over 10000 warriors under Kaom\n"))
			-- The timeless LUTs don't load headless, so simulate a conquest on an allocated node in radius
			local item = build.itemsTab.items[build.spec.jewels[socket.id]]
			local conquered
			for nodeId in pairs(socket.nodesInRadius[item.jewelRadiusIndex]) do
				local node = build.spec.nodes[nodeId]
				if node and node.alloc and node.type == "Normal" then
					conquered = node
					break
				end
			end
			assert.is_truthy(conquered)
			conquered.conqueredBy = { id = 10000, conqueror = { id = 1, type = "karui" } }
			conquered.modList = new("ModList"):ModList()
			conquered.modList:NewMod("Life", "BASE", 100, "Timeless Jewel")
			build.calcsTab:BuildOutput()

			local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat })
			assert.are.equal(socket.id, optimizer.conqueredBySocket[conquered.id])
			assert.is_nil(optimizer.mustHave[socket.id])
			local state = currentState(optimizer)
			local without = optimizer:CopyState(state)
			optimizer:RemoveNode(without, socket.id)
			assert.is_true(optimizer:Evaluate(state).value - optimizer:Evaluate(without).value >= 100)
			-- The spec's conquered node is restored after evaluating
			assert.are.equal(conquered, build.spec.nodes[conquered.id])
		end)

		it("counts nodes in an Intuitive Leap-like radius as connected while the socket is in the tree", function()
			local socket = socketJewel(new("Item"):Item("Rarity: UNIQUE\nThread of Hope\nCrimson Jewel\nVariant: Large Ring\nSelected Variant: 1\nRadius: Variable\nImplicits: 0\nOnly affects Passives in Large Ring\nPassives in Radius can be Allocated without being connected to your Tree\n"))
			local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat })
			local target
			for _, node in ipairs(optimizer.leapNodes) do
				if not node.alloc and node.type == "Normal" and optimizer:IsEligible(node) then
					target = node
					break
				end
			end
			assert.is_truthy(target)
			local state = currentState(optimizer)
			local dist = optimizer:Frontier(state, 4)
			assert.are.equal(1, dist[target.id])

			optimizer:AddNode(state, target)
			assert.is_true(optimizer:Reachable(state.nodes)[target.id] == true)
			-- Without the socket the node is cut off
			assert.is_nil(optimizer:Reachable(state.nodes, socket.id)[target.id])
		end)
	end)

	describe("stopping while trimming", function()
		-- A stopped run that is still over budget must say it was stopped, not that must-have nodes don't fit
		local function overBudgetState(optimizer)
			local state = optimizer:NewState()
			local added = 0
			for _, node in pairs(build.spec.nodes) do
				if node.type == "Normal" and not node.ascendancyName and added < 4 then
					optimizer:AddNode(state, node)
					added = added + 1
				end
			end
			return state
		end

		it("reports a stop instead of a budget error", function()
			local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat, points = 1 })
			local state = overBudgetState(optimizer)
			optimizer.stopped = true
			local ok, err = optimizer:Trim(state, 1)
			assert.is_false(ok)
			assert.is_truthy(err:match("^Stopped"))
		end)

		it("still reports the budget error when not stopped", function()
			local optimizer = TreeOptimizer.new(build, { powerStat = lifeStat, points = 1 })
			local state = optimizer:NewState()
			local notable = findNotable(2)
			optimizer:AddNode(state, notable)
			-- the only node is a must-have, so nothing is removable
			optimizer.mustHave = { [notable.id] = true }
			local ok, err = optimizer:Trim(state, 0)
			assert.is_false(ok)
			assert.is_truthy(err)
			assert.is_nil(err:match("^Stopped"))
		end)
	end)

	it("toggles optimizer marks, ignoring ascendancy and start nodes", function()
		local notable = assert(findNotable(2))
		local treeTab = build.treeTab
		treeTab:ToggleOptimizerMark(notable, "must")
		assert.are.equal("must", build.spec.optimizerMarks[notable.id])
		treeTab:ToggleOptimizerMark(notable, "not")
		assert.are.equal("not", build.spec.optimizerMarks[notable.id])
		treeTab:ToggleOptimizerMark(notable, "not")
		assert.is_nil(build.spec.optimizerMarks[notable.id])

		for _, node in pairs(build.spec.nodes) do
			if node.ascendancyName or node.type == "ClassStart" then
				treeTab:ToggleOptimizerMark(node, "must")
				assert.is_nil(build.spec.optimizerMarks[node.id])
				break
			end
		end
	end)

	it("saves and loads optimizer marks and requirements", function()
		local mustHave = assert(findNotable(3))
		local mustNot = assert(findNotable(5))
		build.spec.optimizerMarks[mustHave.id] = "must"
		build.spec.optimizerMarks[mustNot.id] = "not"
		build.spec.optimizerRequirements.Str = 120
		build.spec.optimizerRequirements.Int = 80
		build.spec.optimizerRequirements.ManaUnreserved = 150
		build.spec.optimizerRequirements.SustainCost = 0

		loadBuildFromXML(build:SaveDB("test"), "test")
		assert.are.equal(0, build.spec.optimizerRequirements.SustainCost)
		assert.are.equal("must", build.spec.optimizerMarks[mustHave.id])
		assert.are.equal("not", build.spec.optimizerMarks[mustNot.id])
		assert.are.equal(120, build.spec.optimizerRequirements.Str)
		assert.are.equal(80, build.spec.optimizerRequirements.Int)
		assert.is_nil(build.spec.optimizerRequirements.Dex)
		assert.are.equal(150, build.spec.optimizerRequirements.ManaUnreserved)
	end)

	it("runs from the Tree tab popup and applies the result as a new tree", function()
		local treeTab = build.treeTab
		local mustHave = assert(findNotable(4))
		build.spec.optimizerMarks[mustHave.id] = "must"
		local oldSpec = build.spec
		local specCount = #treeTab.specList

		treeTab:OpenOptimizerPopup()
		local popup = main.popups[1]
		local controls = popup.controls
		treeTab.optimizerSettings.powerStat = lifeStat
		treeTab.optimizerSettings.points = 10
		treeTab.optimizerSettings.fromScratch = true
		treeTab.optimizerSettings.timeLimit = 2
		assert.is_string(controls.status.label())
		assert.is_falsy(controls.apply:IsEnabled())

		controls.run.onClick()
		local frames = 0
		while treeTab.optimizerRun.co and frames < 10000 do
			treeTab:ResumeOptimizer()
			frames = frames + 1
		end
		local run = treeTab.optimizerRun
		assert.is_nil(run.error)
		assert.is_truthy(run.result)
		assert.is_truthy(run.result.best.nodes[mustHave.id])
		assert.is_string(controls.status.label())
		assert.is_string(controls.details.label())
		assert.is_true(controls.apply:IsEnabled())

		controls.apply.onClick()
		assert.are.equal(specCount + 1, #treeTab.specList)
		assert.are_not.equal(oldSpec, build.spec)
		assert.is_truthy(build.spec.allocNodes[mustHave.id])
		assert.are.equal("must", build.spec.optimizerMarks[mustHave.id])
		-- The optimizer popup closed (other popups may be left over from earlier tests)
		assert.are_not.equal(popup, main.popups[1])
	end)

	it("applies the result as a new tree that matches the prediction", function()
		local optimizer, result = run({ points = 10, fromScratch = true })
		local specCount = #build.treeTab.specList
		optimizer:ApplyToNewSpec(result.best)
		assert.are.equal(specCount + 1, #build.treeTab.specList)
		build.buildFlag = true
		runCallback("OnFrame")
		assert.are.equal(result.best.count, (build.spec:CountAllocNodes()))
		assert.are.near(result.bestEval.value, build.calcsTab.mainOutput.Life, 0.5)
	end)
end)
