from nwpipeline import Pipeline

study_dir = 'w:/NiMBaLWEAR/OND09'
collections = [('0012','01'), ('0014','01'), ('0019','01'), ('0020','01'), ('0023','01'), ('0008','01')]
single_stage = 'activity'

test_nwpl = Pipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
