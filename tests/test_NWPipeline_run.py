from src.nwpipeline import Pipeline

study_dir = 'w:/NiMBaLWEAR/OND09'

collections = [('0095', '01')]
#collections = None
single_stage = None

test_nwpl = Pipeline(study_dir)


test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
