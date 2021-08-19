import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test-HANDDS'
collections = [('060821', '01')]
single_stage = 'crop'

test_nwpl = nwpl.NWPipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, overwrite_header=True, gait_axis=0, quiet=True, log=True)
