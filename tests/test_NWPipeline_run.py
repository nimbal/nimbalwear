import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test-HANDDS'
collections = None
single_stage = None

test_nwpl = nwpl.Pipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
