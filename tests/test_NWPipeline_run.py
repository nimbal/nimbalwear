from nwpipeline import Pipeline

study_dir = 'w:/NiMBaLWEAR/dev-OND09'
collections = [('0001','01')]
single_stage = 'activity'

test_nwpl = Pipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
