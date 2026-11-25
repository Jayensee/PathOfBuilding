describe("TestMiscCalculator", function()
	before_each(function()
		newBuild()
	end)


	-- ConfigTab tooltips build a misc calculator once and call it again after changing the config, so the
	-- closure must not freeze anything that comes from the config
	it("reflects config changes made after the calculator was created", function()
		build.configTab.input.customMods = ""
		build.configTab.input.bandit = "None"
		build.configTab.input.resistancePenalty = nil
		build.configTab:BuildModList()
		runCallback("OnFrame")
		local calc = build.calcsTab:GetMiscCalculator()
		local baseLife = calc({}, false).Life
		local baseFire = calc({}, false).FireResist

		build.configTab.input.customMods = "+1000 to maximum Life"
		build.configTab:BuildModList()
		assert.are.equals(baseLife + 1000, calc({}, false).Life)

		build.configTab.input.customMods = ""
		build.configTab.input.bandit = "Oak"
		build.configTab:BuildModList()
		assert.are.equals(baseLife + 40, calc({}, false).Life)

		build.configTab.input.bandit = "None"
		build.configTab.input.resistancePenalty = 0
		build.configTab:BuildModList()
		assert.are.equals(baseFire + 60, calc({}, false).FireResist)
	end)
end)
