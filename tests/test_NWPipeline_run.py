from src.nwpipeline import Pipeline

study_dir = 'w:/NiMBaLWEAR/OND08'

collections = [('0001', '01')]
collections = None
single_stage = 'activity'

test_nwpl = Pipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
