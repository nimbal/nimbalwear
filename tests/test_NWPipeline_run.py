import nwpipeline as nwpl

study_dir = 'w:/NiMBaLWEAR/dev-OND09'
collections = [('0007','01')]
single_stage = 'sleep'

test_nwpl = nwpl.Pipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
