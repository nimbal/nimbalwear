from nwpipeline import Pipeline

study_dir = 'w:/dev/dev-nimbalwear/dev-OND09'

collections = [('0001', '01')]
single_stage = 'gait'

test_nwpl = Pipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
