import nwpipeline as nwpl

study_dir = 'w:/NiMBaLWEAR/test-HANDDS'
collections = [('060821','01')]
single_stage = 'gait'

test_nwpl = nwpl.Pipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
